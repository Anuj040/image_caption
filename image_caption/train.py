"""model definition, train procedure and the essentials"""
import datetime
import os
import re
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

from image_caption.models import (
    CNNModel,
    Transformer,
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
        image_size = (256, 224)
        train_transform = transforms.Compose(
            [
                # transforms.Resize((356, 356)),
                # transforms.RandomCrop(image_size),
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
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
        self.loss_weights = torch.Tensor(train_dataset.vocab.weights)

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
            self.text_embed_size = self.image_embed_size
            return vocab_size, train_loader, valid_loader, None

        embedding_matrix, self.text_embed_size = prepare_embeddings(
            "datasets/token_embeds", train_dataset.vocab, self.image_embed_size
        )
        return vocab_size, train_loader, valid_loader, embedding_matrix

    def get_current_state(self, path: str) -> Tuple[int, dict, dict]:
        """reloads the current state dict for models and auxillaries

        Args:
            path (str): path to the checkpoint

        Returns:
            Tuple[int, dict, dict]: current epoch number, optimizer and scheduler states
        """
        print(f"=> loading checkpoint '{path}'")
        checkpoint = torch.load(path, map_location=torch.device(DEVICE))
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        return checkpoint["epoch"], checkpoint["optim_state"], checkpoint["scheduler"]

    # pylint: disable = too-many-arguments, too-many-statements
    def train(
        self,
        seq_length: int = 25,
        epochs: int = 10,
        batch_size: int = 64,
        num_workers: int = 8,
        reload_path: Optional[str] = None,
    ) -> None:
        """preparation of model definitions and execute train/valid step

        Args:
            seq_length (int, optional): Fixed length allowed for any sequence. Defaults to 25.
            epochs (int, optional): Total epochs to run. Defaults to 10.
            batch_size (int, optional): Defaults to 64.
            num_workers (int, optional): Defaults to 8.
            reload_path (Optional[str], optional): Checkpoint path, if provided, model
                    train will resume from the checkpoint. Defaults to None.
        """
        vocab_size, train_loader, valid_loader, embedding_matrix = self.generators(
            seq_length, batch_size, num_workers
        )

        # Model definitions
        cnn_model: nn.Module = CNNModel(
            img_embd_size=self.input_embed_size, trainable=self.trainable
        ).to(DEVICE)
        self.transformer = Transformer(
            input_embed_size=self.input_embed_size, vocab_size=vocab_size
        ).to(DEVICE)

        # Prepare the optimizer & loss functions
        lrate = 1e-4 * batch_size / 64

        optimizer = Adam(self.transformer.parameters(), lr=1e-6)
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer,
            anneal_strategy="cos",
            anneal_epochs=2,
            swa_lr=lrate,
        )
        plt_scheduler = ReduceLROnPlateau(optimizer, "max", factor=0.5, patience=5)
        scheduler_switch_epoch = 15
        if reload_path is not None:
            # Resume from checkpoint
            start_epoch, optim_state, scheduler_state = self.get_current_state(
                reload_path
            )
            optimizer.load_state_dict(optim_state)
            if start_epoch > scheduler_switch_epoch:
                swa_scheduler.load_state_dict(scheduler_state)
            else:
                plt_scheduler.load_state_dict(scheduler_state)

            # Logging and checkpoints
            now = os.path.basename(os.path.dirname(reload_path))
            writer = SummaryWriter(f"log/{now}")

            loss_regex = r"-(\d*.\d{4})"
            matches = re.search(loss_regex, os.path.basename(reload_path))

            # Initialize validation loss from saved state
            valid_loss = float(matches.group(1))

        else:
            start_epoch = 0
            # Logging and checkpoints
            now = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
            os.makedirs(f"transformer/checkpoint/{now}", exist_ok=True)
            writer = SummaryWriter(f"transformer/log/{now}")

            # Initialize validation loss
            valid_loss = -1e9

        scaler = GradScaler(enabled=torch.cuda.is_available())

        # Run loop
        cnn_model.eval()
        for epoch in range(start_epoch, epochs):

            self.transformer.train()
            scheduler = (
                swa_scheduler if epoch < scheduler_switch_epoch else plt_scheduler
            )
            print(
                f"========= Runnnig epoch {epoch+1:04d} of {epochs:04d} epochs. ========="
            )
            for i, (imgs, captions, _) in enumerate(tqdm(train_loader)):
                step = epoch * len(train_loader) + i + 1
                imgs = imgs.to(DEVICE)
                img_embed = cnn_model(imgs)

                batch_loss = 0.0
                batch_acc = 0.0
                for caption in captions:
                    optimizer.step()
                    with autocast(enabled=torch.cuda.is_available()):
                        loss, acc = self._compute_caption_loss_and_acc(
                            img_embed, caption.to(DEVICE), self.loss_weights
                        )
                    scaler.scale(loss).backward(retain_graph=True)
                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad()
                    batch_loss += loss
                    batch_acc += acc

                del img_embed, imgs, captions

                if (i + 1) % int(50 * 8 / batch_size) == 0:
                    writer.add_scalar("loss", batch_loss / self.num_captions, step)
                    writer.add_scalar("acc", batch_acc / self.num_captions, step)
            # Evaluation step
            current_val_loss = self.valid(
                cnn_model, valid_loader, writer, epoch + 1, vocab_size
            )

            if current_val_loss > valid_loss:
                valid_loss = current_val_loss
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model": self.transformer.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    f"transformer/checkpoint/{now}/model-{epoch+1:04d}-{valid_loss:.4f}.pth",
                )
            # Update l_rates
            scheduler.step() if epoch < scheduler_switch_epoch else scheduler.step(
                valid_loss
            )

    @staticmethod
    def calculate_loss(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask: torch.Tensor,
        loss_weights: torch.Tensor = None,
        mode: str = "train",
    ) -> torch.Tensor:
        """calculates the error between prediction and ground truth

        Args:
            y_true (torch.Tensor): target sequence
            y_pred (torch.Tensor): predicted sequence prob. matrix (N * vocab_size * seq_len)
            mask (torch.Tensor):
            loss_weights (torch.Tensor): class weight for each token dependent on their occurence
                        frequency
            mode (str, optional): Determines the loss fn. to use. Defaults to "train".

        Returns:
            torch.Tensor: loss value
        """
        mask = mask.to(dtype=float)

        one_hot_true = (
            F.one_hot(y_true, num_classes=y_pred.size(1))
            .transpose(-2, -1)
            .to(dtype=torch.float32)
        )
        if mode == "valid":
            # Without weighing different tokens differently
            loss = one_hot_true * torch.log(y_pred) + (1 - one_hot_true) * torch.log(
                1 - y_pred
            )
            loss = -loss * mask.unsqueeze(1)
            return torch.sum(loss) / torch.sum(mask)

        alpha = 0.25
        gamma = 4.0
        smooth = 0.1
        loss = (
            # loss_weights.to(DEVICE)
            # *
            one_hot_true * torch.log(y_pred + smooth)  # * (1 - y_pred) ** gamma
            # + (1 - alpha)
            + (1 - one_hot_true) * torch.log(1 - y_pred)  # * (y_pred) ** gamma
        )
        # loss = (
        #     -loss * loss_weights.to(DEVICE) * mask.unsqueeze(1)
        # )
        loss = -loss * mask.unsqueeze(1)
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
        # First token is often "a", so ignoring it to get better idea for acc.
        y_true, y_pred, mask = y_true[..., 1:], y_pred[..., 1:], mask[..., 1:]

        accuracy = torch.eq(y_true, torch.argmax(y_pred, axis=1))
        accuracy = torch.logical_and(mask, accuracy).to(float)
        return torch.sum(accuracy) / torch.sum(mask.to(float))

    def _compute_caption_loss_and_acc(
        self,
        img_embed: torch.Tensor,
        batch_seq: torch.Tensor,
        loss_weights: torch.Tensor = None,
        mode: str = "train",
    ) -> Tuple[torch.Tensor]:
        """
            1. uses image embeddings to predict the caption
            2. prepares the mask for error and acc calculation
            3. error and acc calc.

        Args:
            img_embed (torch.Tensor): image embeddings from feature extractor netowrk
            batch_seq (torch.Tensor): ground truth sequence (captions)
            loss_weights (torch.Tensor): class weight for each token dependent on their occurence
                        frequency
            mode (str, optional): Determines the loss fn. to use. Defaults to "train".

        Returns:
            Tuple[torch.Tensor]: loss/acc values
        """

        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:].long()

        # Ignore tokens <UNK>, <PAD>, etc.
        enc_mask = torch.not_equal(batch_seq_inp, self.ignore_indices[0]).to(float)
        dec_mask = torch.not_equal(batch_seq_true, self.ignore_indices[0]).to(float)
        for token_index in self.ignore_indices[1:]:
            temp_mask = torch.not_equal(batch_seq_inp, token_index).to(float)
            enc_mask = torch.minimum(enc_mask, temp_mask)
            temp_mask = torch.not_equal(batch_seq_true, token_index).to(float)
            dec_mask = torch.minimum(dec_mask, temp_mask)

        batch_seq_pred, _, _ = self.transformer(
            img_embed, batch_seq_inp, mask=enc_mask.to(DEVICE)
        )
        batch_seq_pred = batch_seq_pred.permute(0, 2, 1)
        loss = self.calculate_loss(
            batch_seq_true, batch_seq_pred, dec_mask, loss_weights, mode
        )
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, dec_mask)

        del img_embed, batch_seq, batch_seq_inp, batch_seq_true
        del enc_mask, dec_mask, batch_seq_pred
        return loss, acc

    def valid(
        self,
        cnn_model: nn.Module,
        loader: DataLoader,
        writer: SummaryWriter,
        step: int,
        vocab_size: int,
    ) -> Tuple[float]:
        """model evaluation step executor

        Args:
            cnn_model (nn.Module): image feature extracotor
            dataloader (DataLoader): validation data loader
            writer (SummaryWriter): tensorboard summary writer object
            step (int): train epoch
            vocab_size (int):

        Returns:
            float: validation acc.
        """
        self.transformer.eval()
        loss_total = 0
        loss_count = 0
        acc_mean = 0
        with torch.no_grad():
            for (imgs, captions, _) in loader:
                batch_size = imgs.size(0)
                imgs = imgs.to(DEVICE)

                img_embed = cnn_model(imgs)
                batch_loss = 0.0
                batch_acc = 0.0
                for caption in captions:
                    loss, acc = self._compute_caption_loss_and_acc(
                        img_embed, caption.to(DEVICE), mode="valid"
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
        return acc_mean / loss_count / self.num_captions

    def beam_infer(
        self,
        seq_length: int = 25,
        beam_size: int = 1,
        reload_path: str = None,
    ) -> None:
        """preparation of model definitions and execute train/valid step

        Args:
            seq_length (int, optional): Fixed length allowed for any sequence. Defaults to 25.
            beam_size (int, optional): Top-k predictions to chose. Defaults to 1.
            reload_path (Optional[str], optional): Checkpoint path
        """
        # Data augmentation for image data
        image_size = (224, 224)
        img_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        # Prepare the vocab
        train_dataset = CaptionDataset(seq_length=seq_length)
        vocab = train_dataset.vocab
        vocab_size = len(vocab)

        # Model definitions
        cnn_model: nn.Module = CNNModel(
            img_embd_size=self.input_embed_size, trainable=self.trainable
        ).to(DEVICE)
        transformer: nn.Module = Transformer(
            input_embed_size=self.input_embed_size, vocab_size=vocab_size
        ).to(DEVICE)
        print(f"=> loading checkpoint '{reload_path}'")
        checkpoint = torch.load(reload_path, map_location=torch.device(DEVICE))
        transformer.load_state_dict(checkpoint["model"])

        images = [
            "datasets/Flicker8k_Dataset/1002674143_1b742ab4b8.jpg",
            "datasets/Flicker8k_Dataset/1030985833_b0902ea560.jpg",
        ]

        cnn_model.eval()
        transformer.eval()
        with torch.no_grad():
            for ind, img_path in enumerate(images):
                img = Image.open(img_path).convert("RGB")
                img = img_transform(img).unsqueeze(0)

                # Pass the image to the CNN
                img = cnn_model(img)
                # Pass the image features to the Transformer encoder
                img = transformer.dense_1(transformer.layernorm_1(img))
                encoded_img, _ = transformer.encoder(img)

                # beam_search_index
                k = beam_size
                # encoded_img = encoded_img.tile([k, 1, 1])
                encoded_img = img.tile([k, 1, 1])

                input_seq = torch.Tensor(
                    [[train_dataset.vocab.stoi["<SOS>"]] for _ in range(k)]
                )
                # Tensor to store top k sequences' scores; now they're just 0 # (k, 1)
                top_k_scores = torch.zeros(k, 1).to(DEVICE)
                step = 1
                # Lists to store completed sequences and scores
                complete_seqs = []
                complete_seqs_scores = []
                # Generate the caption using the Transformer decoder
                while True:
                    pred, _ = transformer.decoder(
                        input_seq.to(dtype=torch.int32),
                        encoded_img,
                        None,
                    )
                    pred = transformer.final_layer(pred)
                    # Add # (s, vocab_size)
                    pred = top_k_scores.expand_as(pred[:, -1]) + pred[:, -1]
                    if step == 1:
                        top_k_scores, top_k_words = pred[0].topk(k, -1, True, True)
                    else:
                        pred = pred.reshape(-1)
                        top_k_scores, top_k_words = pred.topk(k, -1, True, True)
                    # Convert unrolled indices to actual indices of scores
                    prev_word_inds = top_k_words / vocab_size  # (s)
                    next_word_inds = top_k_words % vocab_size
                    # Add new words to sequences
                    input_seq = torch.cat(
                        [input_seq[prev_word_inds.long()], next_word_inds.unsqueeze(1)],
                        dim=1,
                    )
                    # Check for incomplete (didn't reach <EOS>) sequences
                    incomplete_inds = [
                        ind
                        for ind, next_word in enumerate(next_word_inds)
                        if next_word != vocab.stoi["<EOS>"]
                    ]
                    complete_inds = list(set(range(k)) - set(incomplete_inds))
                    # Set aside complete sequences
                    if len(complete_inds) > 0:
                        complete_seqs.extend(input_seq[complete_inds].tolist())
                        complete_seqs_scores.extend(top_k_scores[complete_inds])

                    # reduce beam length accordingly
                    k -= len(complete_inds)
                    # Proceed with incomplete sequences
                    if k == 0:
                        break
                    input_seq = input_seq[incomplete_inds]
                    encoded_img = encoded_img[incomplete_inds]
                    top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                    # Break if things have been going on too long
                    if step >= seq_length:
                        break
                    step += 1
                if complete_seqs_scores:
                    i = complete_seqs_scores.index(max(complete_seqs_scores))
                    seq = complete_seqs[i]
                else:
                    seq = input_seq[0].tolist()
                seq = (
                    " ".join(vocab.itos[token_index] for token_index in seq[1:-1]) + "."
                )
                print(f"Predicted Caption {ind}: ", seq)


if __name__ == "__main__":  # pragma: no cover
    model = Caption(trainable=False, use_pretrained=False, use_alibi=False)
    MODEL_PATH = "transformer/checkpoint/16112021_210756/model-0005-0.2683.pth"
    # model.train(
    #     seq_length=25,
    #     epochs=40,
    #     batch_size=64,
    #     num_workers=4,
    #     reload_path=MODEL_PATH,
    # )
    model.beam_infer(
        seq_length=25,
        beam_size=3,
        reload_path=MODEL_PATH,
    )
