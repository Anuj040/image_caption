"""model definition, train procedure and the essentials"""
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from image_caption.models import (
    CNNModel,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)
from image_caption.utils.generator import CaptionDataset, Collate

# check if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Caption:
    """object for preparing model definition, train procedure and the essentials"""

    def __init__(
        self,
        trainable: bool,
    ) -> None:
        # Dimension for the image embeddings and token embeddings
        self.image_embed_size: int = 512
        # Per-layer units in the feed-forward network
        self.ff_dim: int = 512
        # Heads for multihead attention for encoder network
        self.num_heads: int = 1
        # Emebedding size for the feature extractor output
        self.input_embed_size: int = 1280

        self.trainable = trainable

    @staticmethod
    def generators(
        seq_length: int, batch_size: int, num_workers: int = 8
    ) -> Tuple[int, DataLoader, DataLoader]:
        """prepares data loader objects for model training and evaluation

        Args:
            seq_length (int): [description]
            batch_size (int): [description]
            num_workers (int, optional): [description]. Defaults to 8.

        Returns:
            Tuple[int, DataLoader, DataLoader]: [description]
        """
        # Data augmentation for image data
        train_transform = transforms.Compose(
            [
                transforms.Resize((356, 356)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        valid_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # Prepare the dataloaders
        train_dataset = CaptionDataset(seq_length=seq_length, transform=train_transform)
        vocab_size = len(train_dataset.vocab)
        pad_value = train_dataset.vocab.stoi["<PAD>"]

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=Collate(pad_value),
        )
        valid_dataset = CaptionDataset(
            seq_length=seq_length, transform=valid_transform, split="valid"
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=32,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=Collate(pad_value),
        )
        return vocab_size, train_loader, valid_loader

    def train(
        self, seq_length: int = 25, batch_size: int = 64, num_workers: int = 8
    ) -> None:
        """preparation of model definitions and execute train/valid step

        Args:
            seq_length (int, optional): Fixed length allowed for any sequence. Defaults to 25.
        """
        vocab_size, train_loader, valid_loader = self.generators(
            seq_length, batch_size, num_workers
        )

        # Model definitions
        cnn_model: nn.Module = CNNModel(trainable=self.trainable).to(DEVICE)
        encoder: nn.Module = TransformerEncoderBlock(
            self.image_embed_size,
            num_heads=self.num_heads,
            input_embed_size=self.input_embed_size,
        ).to(DEVICE)
        decoder: nn.Module = TransformerDecoderBlock(
            vocab_size,
            seq_length,
            self.image_embed_size,
            self.ff_dim,
            2 * self.num_heads,
        ).to(DEVICE)

        # for epoch in range(num_epochs):
        #     if save_model:
        #         checkpoint = {
        #             "state_dict": model.state_dict(),
        #             "optimizer": model.state_dict(),
        #             "step": step,
        #         }
        #         save_checkpoint(checkpoint)

        #     #     for idx, (imgs, captions) in tqdm(
        #     #         enumerate(loader), total=len(loader), leave=False
        #     #     ):
        #     for idx, (imgs, captions) in enumerate(loader):
        #         imgs = imgs.to(device)
        #         captions = captions.to(device)

        #         score = model(imgs, captions[:-1])

        #         #         print(score.shape, captions.shape)
        #         #         print(score.reshape(-1, score.shape[2]).shape, captions.reshape(-1).shape)
        #         #         print("why are we reshaping it here?")
        #         optimizer.zero_grad()
        #         loss = loss_criterion(
        #             score.reshape(-1, score.shape[2]), captions.reshape(-1)
        #         )

        #         step += 1

        #         loss.backward()
        #         optimizer.step()
        #     print(f"Loss for epoch {epoch}: {loss}")


if __name__ == "__main__":  # pragma: no cover
    model = Caption(trainable=False)
