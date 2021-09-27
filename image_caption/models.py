"""model architecures"""

import torch
from efficientnet_pytorch import EfficientNet
from torch import nn


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
        self.output_embed_size = output_embed_size
        self.num_heads = num_heads

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
            need_weights=True,
            attn_mask=None,
        )
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1


if __name__ == "__main__":  # pragma: no cover
    cnn = CNNModel()
    encoder = TransformerEncoderBlock(512, 1)
    a = torch.rand((3, 3, 299, 299))

    b = cnn(a)
    c = encoder(b)
    print(c.detach().numpy().shape)
