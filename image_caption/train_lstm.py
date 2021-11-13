"""https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py"""
import datetime
import os
import re
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image
from torch import nn
from torch.backends import cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import transforms

from image_caption.utils.data_utils import prepare_embeddings
from image_caption.utils.generator import CaptionDataset, Collate
from image_caption.utils.model_utils import (
    AverageMeter,
    accuracy,
    adjust_learning_rate,
    clip_gradient,
    save_checkpoint,
)
from models_lstm import DecoderWithAttention, Encoder

# check if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 10  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
num_workers = 4  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.0  # clip gradients at an absolute value of
# regularization parameter for 'doubly stochastic attention', as in the paper
alpha_c = 1.0
best_bleu4 = 0.0  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = (
    None  # "lstm_logs/ACC_67.180_BLEU_0.091.tar"  # path to checkpoint, None if none
)
image_embed_size: int = 300
num_captions = 5


def generators(
    seq_length: int, batch_size: int, num_workers: int = 8
) -> Tuple[Dict, DataLoader, DataLoader, Optional[np.ndarray]]:
    """prepares data loader objects for model training and evaluation

    Args:
        seq_length (int): Fixed length allowed for any sequence.
        batch_size (int): [description]
        num_workers (int, optional): [description]. Defaults to 8.

    Returns:
        Dict: vocabulary
        DataLoader, DataLoader: Train and validation dataset loaders
        Optional[np.ndarray]: pretrained Embedding matrix
    """
    # Data augmentation for image data
    image_size = (224, 224)
    train_transform = transforms.Compose(
        [
            # transforms.Resize((356, 356)),
            # transforms.RandomCrop(image_size),
            transforms.Resize(image_size),
            # transforms.RandomHorizontalFlip(p=0.5),
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
    # Get the vocabulary
    vocab = train_dataset.vocab
    loss_weights = torch.Tensor(train_dataset.vocab.weights)

    ignore_indices = [vocab.stoi["<PAD>"], vocab.stoi["<UNK>"]]
    pad_value = vocab.stoi["<PAD>"]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=Collate(pad_value, num_captions=5),
    )
    valid_dataset = CaptionDataset(
        seq_length=seq_length, transform=valid_transform, split="valid"
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=Collate(pad_value, num_captions=5),
    )
    # if not self.use_pretrained:
    #     self.text_embed_size = self.image_embed_size
    #     return vocab_size, train_loader, valid_loader, None

    embedding_matrix, text_embed_size = prepare_embeddings(
        "datasets/token_embeds", train_dataset.vocab, image_embed_size
    )
    return vocab, train_loader, valid_loader, embedding_matrix


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, vocab
    vocab, train_loader, valid_loader, embedding_matrix = generators(
        seq_length=25, batch_size=batch_size, num_workers=num_workers
    )
    vocab_size = len(vocab)
    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(
            attention_dim=attention_dim,
            embed_dim=emb_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
            dropout=dropout,
        )
        decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=decoder_lr,
        )
        encoder = Encoder(encoded_image_size=7)
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = (
            torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr,
            )
            if fine_tune_encoder
            else None
        )

    else:
        print(f"Loading trained model from {checkpoint}.")
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        epochs_since_improvement = checkpoint["epochs_since_improvement"]
        best_bleu4 = checkpoint["bleu-4"]
        decoder = checkpoint["decoder"]
        decoder_optimizer = checkpoint["decoder_optimizer"]
        encoder = checkpoint["encoder"]
        encoder_optimizer = checkpoint["encoder_optimizer"]
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr,
            )

    # Move to GPU, if available
    decoder = decoder.to(DEVICE)
    encoder = encoder.to(DEVICE)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if no improvement for 8 consecutive epochs, and
        # terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(
            train_loader=train_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch,
        )

        # One epoch's validation
        recent_bleu4 = validate(
            val_loader=valid_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
        )

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print(f"\nEpochs since last improvement: {epochs_since_improvement}\n")
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(
            "Flicker8k_Dataset",
            epoch,
            epochs_since_improvement,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            recent_bleu4,
            is_best,
        )


def train(
    train_loader: DataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
):
    """[summary]

    Args:
        train_loader ([type]): DataLoader for training data
        encoder ([type]): encoder model
        decoder ([type]): decoder model
        criterion ([type]): loss layer
        encoder_optimizer ([type]): optimizer to update encoder's weights (if fine-tuning)
        decoder_optimizer ([type]): optimizer to update decoder's weights
        epoch ([type]): epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, captions, captionslens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(DEVICE)
        for cap_index in range(num_captions):
            caps, caplens = captions[cap_index], captionslens[cap_index]
            caps = caps.to(DEVICE)
            caplens = caplens.to(DEVICE)

            # Forward prop.
            imgs_encode = encoder(imgs)
            scores, caps_sorted, decode_lens, alphas, sort_ind = decoder(
                imgs_encode, caps, caplens
            )

            # We decode starting with <SOS>, targets are all words after <SOS> -> <EOS>
            targets = caps_sorted[:, 1:].long()

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lens, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lens, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, grad_clip)

            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lens))
            top5accs.update(top5, sum(decode_lens))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if (i * num_captions + cap_index) % print_freq == 0:
                print(
                    f"Epoch: [{epoch}][{i * num_captions + cap_index}/{len(train_loader)}]\t"
                    f"Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})"
                )


def validate(val_loader: DataLoader, encoder: nn.Module, decoder: nn.Module, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    # eval mode (no dropout or batchnorm)
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = []  # references (true captions) for calculating BLEU-4 score
    hypotheses = []  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, captions, captionslens) in enumerate(val_loader):
            # Move to GPU, if available
            imgs = imgs.to(DEVICE)
            reference_caps = []
            for cap_index in range(num_captions):
                caps, caplens = captions[cap_index], captionslens[cap_index]
                caps = caps.to(DEVICE)
                caplens = caplens.to(DEVICE)

                # Forward prop.
                if encoder is not None:
                    imgs_encode = encoder(imgs)
                scores, caps_sorted, decode_lens, alphas, sort_ind = decoder(
                    imgs_encode, caps, caplens
                )

                # We decode starting with <SOS>, targets are all words after <SOS> -> <EOS>
                targets = caps_sorted[:, 1:].long()

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                scores = pack_padded_sequence(
                    scores, decode_lens, batch_first=True
                ).data
                targets = pack_padded_sequence(
                    targets, decode_lens, batch_first=True
                ).data

                # Calculate loss
                loss = criterion(scores, targets)

                # Add doubly stochastic attention regularization
                loss += alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

                # Keep track of metrics
                losses.update(loss.item(), sum(decode_lens))
                top5 = accuracy(scores, targets, 5)
                top5accs.update(top5, sum(decode_lens))
                batch_time.update(time.time() - start)

                start = time.time()

                if (i * num_captions + cap_index) % print_freq == 0:
                    print(
                        f"Validation: [{i * num_captions + cap_index}/{len(val_loader)}]\t"
                        f"Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                        f"Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})\t"
                    )
                for index, caption in enumerate(caps.tolist()):
                    if cap_index == 0:
                        reference_caps.append([caption])
                    else:
                        reference_caps[index].append(caption)
            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References # because images were sorted in the decoder
            reference_caps = [
                x for _, x in sorted(zip(sort_ind.tolist(), reference_caps))
            ]
            for img_caps in reference_caps:
                # remove <SOS> and pads
                img_captions = list(
                    map(
                        lambda c: [
                            w
                            for w in c
                            if w not in (vocab.stoi["<SOS>"], vocab.stoi["<PAD>"])
                        ],
                        img_caps,
                    )
                )
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            # remove pads
            temp_preds = [preds[j][: decode_lens[j]] for j, p in enumerate(preds)]
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            f"\n * LOSS - {losses.avg:.3f}, TOP-5 ACCURACY - {top5accs.avg:.3f}, BLEU-4 - {bleu4}\n"
        )

    return bleu4


if __name__ == "__main__":
    main()
