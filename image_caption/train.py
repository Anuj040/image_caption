"""model definition, train procedure and the essentials"""
import torch

from image_caption.models import CNNModel

# check if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Caption:
    """object for preparing model definition, train procedure and the essentials"""

    def __init__(self, trainable: bool) -> None:
        self.cnn_model = CNNModel(trainable=trainable).to(DEVICE)


if __name__ == "__main__":  # pragma: no cover
    model = Caption(trainable=False)
