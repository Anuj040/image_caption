"""
model architecures
https://keras.io/examples/vision/image_captioning/#building-the-model
"""

from typing import Tuple

import torch
from efficientnet_pytorch import EfficientNet
from torch import nn

from image_caption.utils.model_utils import get_alibi_mask

# check if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# pylint: disable = useless-super-delegation, too-many-instance-attributes, too-many-arguments
class Identity(nn.Module):
    """Custom module for replacing unnecessry layers from pretrained models"""

    def __init__(self):
        """Initialize"""
        super().__init__()

    @staticmethod
    def forward(inputs: torch.Tensor) -> torch.Tensor:
        "forward pass"
        return inputs


class CNNModel(nn.Module):

    """Feature extractor model for image embeddings"""

    def __init__(self, img_embd_size, trainable: bool = False) -> None:
        """initialize

        Args:
            trainable (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        backbone = EfficientNet.from_pretrained("efficientnet-b0")
        backbone.requires_grad = trainable
        # Remove Unnecessary layers
        backbone._bn1 = Identity()  #
        backbone._avg_pooling = Identity()
        backbone._fc = Identity()
        backbone._swish = Identity()
        sequence = [backbone]
        self.model = nn.Sequential(*sequence)

        self.img_embd_size = img_embd_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Feature extractor's forward function.
            Calls extract_features to extract features, squeezes extra dimensions.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this model after processing.
        """

        # Feature extractor layers
        features = self.model(inputs)
        features = torch.reshape(features, (features.shape[0], -1, self.img_embd_size))
        # features = self.norm(features)
        return features


def point_wise_feed_forward_network(d_model: int, dff: int) -> nn.Module:
    """feed forward layer

    Args:
        d_model (int): embd dimension
        dff (int): sparse dimension

    Returns:
        nn.Module
    """
    return nn.Sequential(
        *[  # (batch_size, seq_len, dff)
            nn.Linear(d_model, dff),
            nn.ReLU(),
            # (batch_size, seq_len, d_model)
            nn.Linear(dff, d_model),
        ]
    )


class EncoderLayer(nn.Module):
    """Basic encoder layer"""

    def __init__(
        self, d_model: int = 512, num_heads: int = 1, dff: int = 2048, rate: float = 0.1
    ) -> None:
        """Initialization"""
        super().__init__()

        self.mha = nn.MultiheadAttention(
            d_model, num_heads, dropout=rate, bias=True, batch_first=True
        )
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor]:
        """forward pass for the encoder layer"""

        # (batch_size, input_seq_len, d_model)
        attn_output, attn_weights = self.mha(
            query=inputs,
            key=inputs,
            value=inputs,
            need_weights=self.training,
            attn_mask=mask,
        )
        attn_output = self.dropout1(attn_output)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(inputs + attn_output)

        # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, attn_weights


class DecoderLayer(nn.Module):
    """Base decoder layer"""

    def __init__(
        self, d_model: int = 512, num_heads: int = 1, dff: int = 2048, rate: float = 0.1
    ) -> None:
        """Initialization"""
        super().__init__()

        self.mha1 = nn.MultiheadAttention(
            d_model, num_heads, dropout=rate, bias=True, batch_first=True
        )
        self.mha2 = nn.MultiheadAttention(
            d_model, num_heads, dropout=rate, bias=True, batch_first=True
        )

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(
        self,
        inputs: torch.Tensor,
        enc_output: torch.Tensor,
        look_ahead_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """forward pass

        Args:
            inputs (torch.Tensor): [description]
            enc_output (torch.Tensor): [description]
            look_ahead_mask (torch.Tensor): causal mask

        Returns:
            Tuple[torch.Tensor]: [description]
        """
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(
            query=inputs,
            key=inputs,
            value=inputs,
            need_weights=self.training,
            attn_mask=look_ahead_mask,
        )
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + inputs)

        # (batch_size, target_seq_len, d_model)
        attn2, attn_weights_block2 = self.mha2(
            query=out1,
            key=enc_output,
            value=enc_output,
            need_weights=self.training,
            attn_mask=None,
        )
        attn2 = self.dropout2(attn2)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)
        # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(nn.Module):
    """Encoder block"""

    def __init__(
        self,
        num_layers: int,
        d_model: int = 512,
        num_heads: int = 1,
        dff: int = 2048,
        rate: float = 0.1,
    ):
        """Initialization"""
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.dropout = nn.Dropout(rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """forward pass"""
        # Image embeddings
        # (batch_size, pixels, d_model)
        embeds = self.dropout(inputs)
        attn_weights = {}
        for i in range(self.num_layers):
            embeds, block = self.enc_layers[i](embeds, mask=None)
            attn_weights[f"encoder_layer{i+1}_block"] = block
        # (batch_size, encoded_pixels, d_model)
        return embeds, attn_weights


class Decoder(nn.Module):
    """Decoder block"""

    def __init__(
        self,
        num_layers: int,
        d_model: int = 512,
        num_heads: int = 1,
        dff: int = 2048,
        vocab_size: int = 10000,
        max_pos_encode: int = 50,
        rate: float = 0.1,
    ):
        """Initialization"""
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = PositionalEmbedding(
            embed_dim=d_model,
            sequence_length=max_pos_encode,
            vocab_size=vocab_size,
            use_alibi=False,
        )

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.dropout = nn.Dropout(rate)

    def forward(
        self,
        inputs: torch.Tensor,
        enc_output: torch.Tensor,
        look_ahead_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """forward pass"""

        attention_weights = {}
        # adding embedding and position encoding.
        # (batch_size, input_seq_len, d_model)
        embeds = self.embedding(inputs)
        embeds = self.dropout(embeds)

        for i in range(self.num_layers):
            embeds, block1, block2 = self.dec_layers[i](
                embeds, enc_output, look_ahead_mask
            )
        attention_weights[f"decoder_layer{i+1}_block1"] = block1
        attention_weights[f"decoder_layer{i+1}_block2"] = block2

        # (batch_size, input_seq_len, d_model)
        return embeds, attention_weights


class Transformer(nn.Module):
    """Transformer"""

    def __init__(
        self,
        num_layers: int = 1,
        d_model: int = 512,
        num_heads: int = 1,
        dff: int = 2048,
        vocab_size: int = 10000,
        max_pos_encode: int = 50,
        rate: float = 0.1,
    ):
        """initialization"""
        super().__init__()
        self.num_heads = num_heads
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)

        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, vocab_size, max_pos_encode, rate
        )

        self.final_layer = nn.Softmax(dim=-1)

    def forward(
        self, imgs: torch.Tensor, tar: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """forward"""
        if mask is not None:
            causal_mask = self.get_causal_attention_mask(tar)
            combined_mask = torch.unsqueeze(mask, 1)
            combined_mask = 1.0 - torch.minimum(combined_mask, causal_mask)
            combined_mask = torch.tile(combined_mask, [self.num_heads, 1, 1]).to(bool)
        else:
            combined_mask = 1.0 - self.get_causal_attention_mask(tar)
            combined_mask = torch.tile(combined_mask, [self.num_heads, 1, 1]).to(bool)

        # (batch_size, #pixels, d_model)
        enc_output, encoder_attns = self.encoder(imgs)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, decoder_attns = self.decoder(tar, enc_output, combined_mask)
        # (batch_size, seq_len, vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output, encoder_attns, decoder_attns

    @staticmethod
    def get_causal_attention_mask(inputs: torch.Tensor) -> torch.Tensor:
        """prepares a mask such that attention is paid only to the ancestors"""
        input_shape = inputs.size()
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = torch.unsqueeze(torch.arange(0, sequence_length), -1)
        j = torch.arange(0, sequence_length)
        mask = (i >= j).to(dtype=float)
        mask = torch.reshape(mask, (1, input_shape[1], input_shape[1]))
        return torch.tile(mask, [batch_size, 1, 1]).to(DEVICE)


class TransformerEncoderBlock(nn.Module):
    """encoder model for transforming image embeddings to one relatable to sequence embeddings"""

    def __init__(
        self,
        output_embed_size: int,
        num_heads: int,
        input_embed_size: int = 1280,
        **kwargs,
    ):
        """initialize

        Args:
            output_embed_size (int): embedding size for the output sequences
            num_heads (int):
            input_embed_size (int, optional): embedding size for input sequences Defaults to 1280.
        """
        super().__init__(**kwargs)
        self.attention_1 = nn.MultiheadAttention(
            output_embed_size, num_heads, dropout=0.0, bias=True, batch_first=True
        )
        self.layernorm_1 = nn.LayerNorm(input_embed_size)
        self.layernorm_2 = nn.LayerNorm(output_embed_size)
        self.dense_1 = nn.Linear(input_embed_size, output_embed_size, bias=True)
        self.act_1 = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """feature encoder's forward pass"""
        inputs = self.act_1(self.dense_1(self.layernorm_1(inputs)))
        attention_output_1, _ = self.attention_1(
            query=inputs,
            key=inputs,
            value=inputs,
            need_weights=False,
            attn_mask=None,
        )
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1


class PositionalEmbedding(nn.Module):
    """converts a given vector (int) representation of a sequence to its embedding representation
    and adds positional embeddings to individual wordvectors
    """

    def __init__(
        self,
        sequence_length: int,
        vocab_size: int,
        embed_dim: int,
        use_alibi: bool = False,
        **kwargs,
    ):
        """initialize

        Args:
            sequence_length (int): sequence lengths
            vocab_size (int): total size of the vocabulary
            embed_dim (int): embedding dimensions
            use_alibi (bool): Use positional embeddings or alibi mask
        """
        super().__init__(**kwargs)
        self.use_alibi = use_alibi
        self.token_embeddings = torch.nn.Embedding(vocab_size, embed_dim)

        if not use_alibi:
            # From train short, test long: https://arxiv.org/abs/2108.12409
            # https://pytorch-lightning.readthedocs.io/en/latest/notebooks/
            # course_UvA-DL/05-transformers-and-MH-attention.html
            self.position_embeddings = torch.nn.Embedding(sequence_length, embed_dim)

        self.embed_scale = torch.sqrt(torch.Tensor([embed_dim])).to(DEVICE)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """forward pass for embedding generator"""
        embedded_tokens = self.token_embeddings(inputs) * self.embed_scale
        if self.use_alibi:
            return embedded_tokens

        length = inputs.shape[-1]
        positions = torch.arange(0, length, step=1).to(DEVICE)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions


class TransformerDecoderBlock(nn.Module):
    """decoder model for translating image embeddings to sequence based on input words"""

    def __init__(
        self,
        vocab_size: int,
        seq_length: int,
        text_embed_dim: int,
        embed_dim: int,
        ff_dim: int,
        num_heads: int,
        use_alibi: bool = False,
    ):
        """initialize

        Args:
            vocab_size (int): total vocabulary size
            seq_length (int): Fixed length allowed for any sequence
            embed_dim (int): dimensions for word embeddings
            ff_dim (int): Per-layer units in the feed-forward network
            num_heads (int): num of attention heads for Multihead attention
            use_alibi (bool): Use positional embeddings or alibi mask
        """
        super().__init__()
        self.num_heads = num_heads
        self.use_alibi = use_alibi

        self.embedding = PositionalEmbedding(
            embed_dim=text_embed_dim,
            sequence_length=seq_length,
            vocab_size=vocab_size,
            use_alibi=use_alibi,
        )
        self.ffn_layer = nn.Linear(text_embed_dim, embed_dim, bias=False)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.attention_1 = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, bias=True, batch_first=True
        )
        self.act_drop_1 = nn.Dropout(0.1)
        self.layernorm_1 = nn.LayerNorm(embed_dim)

        self.attention_2 = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, bias=True, batch_first=True
        )
        self.act_drop_2 = nn.Dropout(0.1)
        self.layernorm_2 = nn.LayerNorm(embed_dim)

        # self.attention_3 = nn.MultiheadAttention(
        #     embed_dim, num_heads, dropout=0.1, bias=True, batch_first=True
        # )
        # self.act_drop_3 = nn.Dropout(0.1)
        # self.layernorm_3 = nn.LayerNorm(embed_dim)

        self.ffn_layer_1 = nn.Linear(embed_dim, ff_dim, bias=True)
        self.act_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.3)

        self.ffn_layer_2 = nn.Linear(ff_dim, embed_dim, bias=False)
        self.layernorm_3 = nn.LayerNorm(embed_dim)

        self.dropout_2 = nn.Dropout(0.5)
        self.out = nn.Linear(embed_dim, vocab_size, bias=True)
        self.out_act = nn.Softmax(dim=-1)

    def forward(self, inputs, encoder_outputs, mask=None):
        """forward pass for the embedding decoder"""
        inputs = self.embedding(inputs)
        # inputs = self.layernorm(self.ffn_layer(inputs))
        padding_mask = None
        if mask is not None:
            padding_mask = 1.0 - torch.unsqueeze(mask, -1)
            padding_mask = torch.tile(padding_mask, [self.num_heads, 1, 1]).to(bool)
            if self.use_alibi:
                causal_mask = get_alibi_mask(inputs, self.num_heads).to(DEVICE)
                combined_mask = 1.0 - torch.unsqueeze(mask, 1)
                new_attn_mask = torch.zeros_like(combined_mask, dtype=torch.float)
                new_attn_mask.masked_fill_(combined_mask.to(bool), float("-inf"))
                new_attn_mask = torch.tile(new_attn_mask, [self.num_heads, 1, 1])
                combined_mask = causal_mask + new_attn_mask
            else:
                causal_mask = self.get_causal_attention_mask(inputs)
                combined_mask = torch.unsqueeze(mask, 1)
                combined_mask = 1.0 - torch.minimum(combined_mask, causal_mask)
                combined_mask = torch.tile(combined_mask, [self.num_heads, 1, 1]).to(
                    bool
                )
        else:
            if self.use_alibi:
                combined_mask = get_alibi_mask(inputs, self.num_heads).to(DEVICE)
            else:
                combined_mask = 1.0 - self.get_causal_attention_mask(inputs)
                combined_mask = torch.tile(combined_mask, [self.num_heads, 1, 1]).to(
                    bool
                )

        attention_output_1, _ = self.attention_1(
            query=inputs,
            key=inputs,
            value=inputs,
            need_weights=False,
            attn_mask=combined_mask,
        )
        out_1 = self.layernorm_1(inputs + self.act_drop_1(attention_output_1))
        attention_output_2, _ = self.attention_2(
            query=out_1,
            key=encoder_outputs,
            value=encoder_outputs,
            need_weights=False,
            attn_mask=None,
        )
        out_2 = self.layernorm_2(out_1 + self.act_drop_2(attention_output_2))

        # attention_output_3, _ = self.attention_3(
        #     query=out_2,
        #     key=encoder_outputs,
        #     value=encoder_outputs,
        #     need_weights=False,
        #     attn_mask=None,
        # )
        # out_3 = self.layernorm_3(out_2 + self.act_drop_3(attention_output_3))
        out_3 = out_2
        ffn_out = self.act_1(self.ffn_layer_1(out_3))
        ffn_out = self.dropout_1(ffn_out)

        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_3)

        ffn_out = self.dropout_2(ffn_out)
        preds = self.out_act(self.out(ffn_out))
        return preds

    @staticmethod
    def get_causal_attention_mask(inputs: torch.Tensor) -> torch.Tensor:
        """prepares a mask such that attention is paid only to the ancestors"""
        input_shape = inputs.size()
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = torch.unsqueeze(torch.arange(0, sequence_length), -1)
        j = torch.arange(0, sequence_length)
        mask = (i >= j).to(dtype=float)
        mask = torch.reshape(mask, (1, input_shape[1], input_shape[1]))
        return torch.tile(mask, [batch_size, 1, 1]).to(DEVICE)


if __name__ == "__main__":  # pragma: no cover
    # Vocabulary size
    VOCAB_SIZE = 10000

    # Fixed length allowed for any sequence
    SEQ_LENGTH = 25

    # Dimension for the image embeddings and token embeddings
    EMBED_DIM = 512

    # Per-layer units in the feed-forward network
    FF_DIM = 512

    # cnn = CNNModel()
    # encoder = TransformerEncoderBlock(EMBED_DIM, 1)
    a = torch.rand((3, 3, 512))

    # b = cnn(a)
    # c = encoder(b)
    # print(c.detach().numpy().shape)

    # decoder = TransformerDecoderBlock(
    #     VOCAB_SIZE, SEQ_LENGTH, EMBED_DIM, EMBED_DIM, FF_DIM, 2
    # )
    # a = torch.randint(0, 10, (3, SEQ_LENGTH))
    # MASK = (a > 4 - 1).to(float)
    # d = decoder(a, c, mask=MASK)
    # print(d.detach().numpy().shape)
    b = torch.randint(0, 10, (3, SEQ_LENGTH))
    dec = Decoder(num_layers=1)
    print(dec(b, a))
