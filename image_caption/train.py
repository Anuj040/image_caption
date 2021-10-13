"""model definition, train procedure and the essentials"""
import datetime
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

from image_caption.models import (
    CNNModel,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)
from image_caption.utils.data_utils import prepare_embeddings
from image_caption.utils.generator import CaptionDataset, Collate

# check if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# pylint: disable = too-many-locals, attribute-defined-outside-init, not-callable
# pylint: disable = too-many-instance-attributes
class Caption:
    """object for preparing model definition, train procedure and the essentials"""

    def __init__(
        self,
        trainable: bool,
        image_embed_size: int = 300,
        use_pretrained: bool = True,
        use_alibi: bool = False,
    ) -> None:
        # number of captions for each image
        self.num_captions: int = 5
        # Dimension for the image embeddings and token embeddings
        self.image_embed_size: int = image_embed_size
        # Whether to use pretrained token embeddings
        self.use_pretrained: bool = use_pretrained
        # Use positional embeddings or alibi mask
        self.use_alibi: bool = use_alibi
        # Per-layer units in the feed-forward network
        self.ff_dim: int = 512
        # Heads for multihead attention for encoder network
        self.num_heads: int = 1
        # Emebedding size for the feature extractor output
        self.input_embed_size: int = 1280

        self.trainable = trainable

    def generators(
        self, seq_length: int, batch_size: int, num_workers: int = 8
    ) -> Tuple[int, DataLoader, DataLoader, Optional[np.ndarray]]:
        """prepares data loader objects for model training and evaluation

        Args:
            seq_length (int): Fixed length allowed for any sequence.
            batch_size (int): [description]
            num_workers (int, optional): [description]. Defaults to 8.

        Returns:
            int: Total size of the vocabulary
            DataLoader, DataLoader: Train and validation dataset loaders
            Optional[np.ndarray]: pretrained Embedding matrix
        """
        # Data augmentation for image data
        image_size = (224, 224)
        train_transform = transforms.Compose(
            [
                transforms.Resize((356, 356)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        valid_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # Prepare the dataloaders
        train_dataset = CaptionDataset(seq_length=seq_length, transform=train_transform)
        vocab_size = len(train_dataset.vocab)

        self.ignore_indices = [
            train_dataset.vocab.stoi["<PAD>"],
            train_dataset.vocab.stoi["<UNK>"],
        ]
        pad_value = train_dataset.vocab.stoi["<PAD>"]

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=Collate(pad_value, self.num_captions),
        )
        valid_dataset = CaptionDataset(
            seq_length=seq_length, transform=valid_transform, split="valid"
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=4,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=Collate(pad_value, self.num_captions),
        )
        if not self.use_pretrained:
            return vocab_size, train_loader, valid_loader, None

        embedding_matrix, self.text_embed_size = prepare_embeddings(
            "datasets/token_embeds", train_dataset.vocab, self.image_embed_size
        )
        return vocab_size, train_loader, valid_loader, embedding_matrix

    def train(
        self,
        seq_length: int = 25,
        epochs: int = 10,
        batch_size: int = 64,
        num_workers: int = 8,
    ) -> None:
        """preparation of model definitions and execute train/valid step

        Args:
            seq_length (int, optional): Fixed length allowed for any sequence. Defaults to 25.
        """
        vocab_size, train_loader, valid_loader, embedding_matrix = self.generators(
            seq_length, batch_size, num_workers
        )

        # Model definitions
        cnn_model: nn.Module = CNNModel(trainable=self.trainable).to(DEVICE)
        self.encoder: nn.Module = TransformerEncoderBlock(
            self.image_embed_size,
            num_heads=self.num_heads,
            input_embed_size=self.input_embed_size,
        ).to(DEVICE)
        self.decoder: nn.Module = TransformerDecoderBlock(
            vocab_size,
            seq_length,
            self.text_embed_size,
            self.image_embed_size,
            self.ff_dim,
            3 * self.num_heads,
            self.use_alibi,
        ).to(DEVICE)

        if self.use_pretrained:
            # Substitute pretrained embeddings for embedding layer
            self.decoder.embedding.token_embeddings.weight = nn.Parameter(
                torch.tensor(embedding_matrix, dtype=torch.float32).to(DEVICE)
            )

        # Prepare the optimizer & loss functions
        lrate = 3e-4
        optimizer = Adam(
            [
                {"params": self.encoder.parameters(), "lr": lrate},
                {"params": self.decoder.parameters(), "lr": lrate},
            ]
        )
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer,
            anneal_strategy="cos",
            anneal_epochs=int(0.2 * epochs),
            swa_lr=lrate,
        )

        scaler = GradScaler(enabled=torch.cuda.is_available())
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

        # Logging and checkpoints
        now = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        os.makedirs(f"checkpoint/{now}", exist_ok=True)
        writer = SummaryWriter(f"log/{now}")

        # Initialize validation loss
        valid_loss = 1e9
        # Run loop
        for epoch in range(0, epochs):

            self.encoder.train()
            self.decoder.train()
            for i, (imgs, captions) in enumerate(tqdm(train_loader)):
                step = epoch * len(train_loader) + i + 1
                imgs = imgs.to(DEVICE)
                img_embed = cnn_model(imgs)

                batch_loss = 0.0
                batch_acc = 0.0
                for caption in captions:
                    optimizer.step()
                    with autocast(enabled=torch.cuda.is_available()):
                        loss, acc = self._compute_caption_loss_and_acc(
                            img_embed, caption.to(DEVICE)
                        )
                    scaler.scale(loss).backward(retain_graph=True)
                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad()
                    swa_scheduler.step()
                    batch_loss += loss
                    batch_acc += acc

                del img_embed, imgs, captions

                if (i + 1) % int(50 * 8 / batch_size) == 0:
                    writer.add_scalar("loss", batch_loss / self.num_captions, step)
                    writer.add_scalar("acc", batch_acc / self.num_captions, step)
            # Evaluation step
            current_val_loss = self.valid(cnn_model, valid_loader, writer, step)

            if current_val_loss < valid_loss:
                valid_loss = current_val_loss
                torch.save(
                    self.encoder.state_dict(),
                    f"checkpoint/{now}/encoder-epoch-{epoch}-loss-{valid_loss:.4f}.pth",
                )
                torch.save(
                    self.decoder.state_dict(),
                    f"checkpoint/{now}/decoder-epoch-{epoch}-loss-{valid_loss:.4f}.pth",
                )

    def calculate_loss(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """calculates the error between prediction and ground truth

        Args:
            y_true (torch.Tensor): target sequence
            y_pred (torch.Tensor): predicted sequence prob. matrix (N * vocab_size * seq_len)
            mask (torch.Tensor):

        Returns:
            torch.Tensor: loss value
        """
        mask = mask.to(dtype=float)
        loss = self.loss_fn(y_pred, y_true) * mask
        return torch.sum(loss) / torch.sum(mask)

    @staticmethod
    def calculate_accuracy(
        y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """calculates the accuracy metric for prediction and ground truth

        Args:
            y_true (torch.Tensor): target sequence
            y_pred (torch.Tensor): predicted sequence prob. matrix (N * vocab_size * seq_len)
            mask (torch.Tensor):

        Returns:
            torch.Tensor: loss value
        """
        accuracy = torch.eq(y_true, torch.argmax(y_pred, axis=1))
        accuracy = torch.logical_and(mask, accuracy).to(float)
        return torch.sum(accuracy) / torch.sum(mask.to(float))

    def _compute_caption_loss_and_acc(
        self, img_embed: torch.Tensor, batch_seq: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
            1. uses image embeddings to predict the caption
            2. prepares the mask for error and acc calculation
            3. error and acc calc.

        Args:
            img_embed (torch.Tensor): image embeddings from feature extractor netowrk
            batch_seq (torch.Tensor): ground truth sequence (captions)

        Returns:
            Tuple[torch.Tensor]: loss/acc values
        """

        encoder_out = self.encoder(img_embed)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:].long()

        # Ignore tokens <UNK>, <PAD>, etc.
        mask = torch.not_equal(batch_seq_true, self.ignore_indices[0]).to(float)
        for token_index in self.ignore_indices[1:]:
            temp_mask = torch.not_equal(batch_seq_true, token_index).to(float)
            mask = torch.minimum(mask, temp_mask)

        batch_seq_pred = self.decoder(batch_seq_inp, encoder_out, mask=mask.to(DEVICE))
        batch_seq_pred = batch_seq_pred.permute(0, 2, 1)
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

        del encoder_out, batch_seq, batch_seq_inp, batch_seq_true
        del mask, batch_seq_pred
        return loss, acc

    def valid(
        self, cnn_model: nn.Module, loader: DataLoader, writer: SummaryWriter, step: int
    ) -> Tuple[float]:
        """model evaluation step executor

        Args:
            cnn_model (nn.Module): image feature extracotor
            dataloader (DataLoader): validation data loader
            writer (SummaryWriter): tensorboard summary writer object
            step (int): train step

        Returns:
            float: validation loss
        """
        self.encoder.eval()
        self.decoder.eval()
        loss_total = 0
        loss_count = 0
        acc_mean = 0
        for (imgs, captions) in loader:
            batch_size = imgs.size(0)
            imgs = imgs.to(DEVICE)

            img_embed = cnn_model(imgs)
            batch_loss = 0.0
            batch_acc = 0.0
            for caption in captions:
                loss, acc = self._compute_caption_loss_and_acc(
                    img_embed, caption.to(DEVICE)
                )
                batch_loss += loss
                batch_acc += acc
            del img_embed, imgs, captions

        loss_total += batch_loss.cpu().item() * batch_size
        acc_mean += batch_acc.cpu().item() * batch_size
        loss_count += batch_size

        writer.add_scalar(
            "valid_loss", loss_total / loss_count / self.num_captions, step
        )
        writer.add_scalar("valid_acc", acc_mean / loss_count / self.num_captions, step)
        return loss_total / loss_count


if __name__ == "__main__":  # pragma: no cover
    model = Caption(trainable=False, use_pretrained=True, use_alibi=False)
    model.train(seq_length=25, epochs=10, batch_size=4)
