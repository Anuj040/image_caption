"""model definition, train procedure and the essentials"""
import torch
from torch import nn

from image_caption.models import CNNModel, TransformerEncoderBlock

# check if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Caption:
    """object for preparing model definition, train procedure and the essentials"""

    def __init__(
        self,
        trainable: bool,
    ) -> None:

        output_embed_size: int = 512
        num_heads: int = 1
        input_embed_size: int = 1280

        self.cnn_model: nn.Module = CNNModel(trainable=trainable).to(DEVICE)
        self.encoder: nn.Module = TransformerEncoderBlock(
            output_embed_size, num_heads=num_heads, input_embed_size=input_embed_size
        ).to(DEVICE)


if __name__ == "__main__":  # pragma: no cover
    model = Caption(trainable=False)
