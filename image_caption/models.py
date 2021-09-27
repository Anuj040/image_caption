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
        features = torch.reshape(features, (-1, features.shape[1]))
        return features


if __name__ == "__main__":  # pragma: no cover
    model = CNNModel()
    a = torch.rand((3, 3, 299, 299))

    b = model(a).detach().numpy()
    print(b.shape)
