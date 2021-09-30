"""
model architecures
https://keras.io/examples/vision/image_captioning/#building-the-model
"""

import torch
from efficientnet_pytorch import EfficientNet
from torch import nn

# check if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CNNModel(nn.Module):

    """Feature extractor model for image embeddings"""

    def __init__(self, trainable: bool = False) -> None:
        """initialize

        Args:
            trainable (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        backbone = EfficientNet.from_name("efficientnet-b0", include_top=False)
        backbone.requires_grad = trainable
        sequence = [backbone]
        self.model = nn.Sequential(*sequence)

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
        features = torch.reshape(features, (features.shape[0], -1, features.shape[1]))
        return features


class TransformerEncoderBlock(nn.Module):
    """encoder model for transforming image embeddings to one relatable to sequence embeddings"""

    def __init__(
        self,
        output_embed_size: int,
        num_heads: int,
        input_embed_size: int = 1280,
        **kwargs
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
        inputs = self.layernorm_1(inputs)
        inputs = self.act_1(self.dense_1(inputs))

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

    def __init__(self, sequence_length: int, vocab_size: int, embed_dim: int, **kwargs):
        """initialize

        Args:
            sequence_length (int): sequence lengths
            vocab_size (int): total size of the vocabulary
            embed_dim (int): embedding dimensions
        """
        super().__init__(**kwargs)
        self.token_embeddings = torch.nn.Embedding(vocab_size, embed_dim)
        #! Try train short, test long: https://arxiv.org/abs/2108.12409
        #! https://pytorch-lightning.readthedocs.io/en/latest/notebooks/
        #! course_UvA-DL/05-transformers-and-MH-attention.html
        self.position_embeddings = torch.nn.Embedding(sequence_length, embed_dim)

        self.embed_scale = torch.sqrt(torch.Tensor([embed_dim])).to(DEVICE)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """forward pass for embedding generator"""
        length = inputs.shape[-1]
        positions = torch.arange(0, length, step=1).to(DEVICE)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = embedded_tokens * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    # def compute_mask(self, inputs, mask=None):
    #     return torch.not_equal(inputs, 0)


# pylint: disable = too-many-instance-attributes, too-many-arguments
class TransformerDecoderBlock(nn.Module):
    """decoder model for translating image embeddings to sequence based on input words"""

    def __init__(
        self,
        vocab_size: int,
        seq_length: int,
        embed_dim: int,
        ff_dim: int,
        num_heads: int,
    ):
        """initialize

        Args:
            vocab_size (int): total vocabulary size
            seq_length (int): Fixed length allowed for any sequence
            embed_dim (int): dimensions for word embeddings
            ff_dim (int): Per-layer units in the feed-forward network
            num_heads (int): num of attention heads for Multihead attention
        """
        super().__init__()
        self.num_heads = num_heads

        self.embedding = PositionalEmbedding(
            embed_dim=embed_dim, sequence_length=seq_length, vocab_size=vocab_size
        )

        self.attention_1 = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, bias=True, batch_first=True
        )
        self.layernorm_1 = nn.LayerNorm(embed_dim)

        self.attention_2 = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, bias=True, batch_first=True
        )
        self.layernorm_2 = nn.LayerNorm(embed_dim)

        self.ffn_layer_1 = nn.Linear(embed_dim, ff_dim, bias=True)
        self.act_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.3)

        self.ffn_layer_2 = nn.Linear(ff_dim, embed_dim, bias=False)
        self.layernorm_3 = nn.LayerNorm(embed_dim)

        self.dropout_2 = nn.Dropout(0.5)
        self.out = nn.Linear(embed_dim, vocab_size, bias=True)
        # self.act_2 = nn.Softmax(dim=-1)
        # self.supports_masking = True

    def forward(self, inputs, encoder_outputs, mask=None):
        """forward pass for the embedding decoder"""
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)

        if mask is not None:
            padding_mask = 1.0 - torch.unsqueeze(mask, -1)
            padding_mask = torch.tile(padding_mask, [self.num_heads, 1, 1]).to(bool)
            combined_mask = torch.unsqueeze(mask, 1)
            combined_mask = 1.0 - torch.minimum(combined_mask, causal_mask)
            combined_mask = torch.tile(combined_mask, [self.num_heads, 1, 1]).to(bool)

        attention_output_1, _ = self.attention_1(
            query=inputs,
            key=inputs,
            value=inputs,
            need_weights=False,
            attn_mask=combined_mask,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2, _ = self.attention_2(
            query=out_1,
            key=encoder_outputs,
            value=encoder_outputs,
            need_weights=False,
            attn_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        ffn_out = self.act_1(self.ffn_layer_1(out_2))
        ffn_out = self.dropout_1(ffn_out)

        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2)

        ffn_out = self.dropout_2(ffn_out)
        preds = self.out(ffn_out)
        # preds = self.act_2(self.out(ffn_out))

        # return preds
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

    cnn = CNNModel()
    encoder = TransformerEncoderBlock(EMBED_DIM, 1)
    a = torch.rand((3, 3, 299, 299))

    b = cnn(a)
    c = encoder(b)
    print(c.detach().numpy().shape)

    decoder = TransformerDecoderBlock(VOCAB_SIZE, SEQ_LENGTH, EMBED_DIM, FF_DIM, 2)
    a = torch.randint(0, 10, (3, SEQ_LENGTH))
    MASK = (a > 4 - 1).to(float)
    d = decoder(a, c, mask=MASK)
    print(d.detach().numpy().shape)
