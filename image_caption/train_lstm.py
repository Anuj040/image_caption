"""https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py"""
import datetime
import os
import re
import time
from typing import Dict, Optional, Tuple, Union

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
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import transforms

from image_caption.models_lstm import DecoderWithAttention, Encoder
from image_caption.utils.data_utils import prepare_embeddings
from image_caption.utils.generator import CaptionDataset, Collate
from image_caption.utils.model_utils import (
    AverageMeter,
    accuracy,
    adjust_learning_rate,
    clip_gradient,
    save_checkpoint,
)

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
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.0  # clip gradients at an absolute value of
# regularization parameter for 'doubly stochastic attention', as in the paper
alpha_c = 1.0
best_bleu4 = 0.0  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
# path to checkpoint, None if none
checkpoint = "BEST_checkpoint_Flicker8k_Dataset.pth.tar"
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


def main(num_workers: int = 4):
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
        decoder_optimizer = Adam(
            params=filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=decoder_lr,
        )
        encoder = Encoder(encoded_image_size=7)
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = (
            Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr,
            )
            if fine_tune_encoder
            else None
        )

    else:
        print(f"Loading trained model from {checkpoint}.")
        checkpoint = torch.load(checkpoint, map_location=torch.device(DEVICE))
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


# pylint: disable = too-many-arguments, too-many-locals
def train(
    train_loader: DataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    criterion: nn.Module,
    encoder_optimizer: Union[Optimizer, None],
    decoder_optimizer: Optimizer,
    epoch: int,
) -> None:
    """training epoch executor

    Args:
        train_loader (DataLoader): DataLoader for training data
        encoder (nn.Module): encoder model
        decoder (nn.Module): decoder model
        criterion (nn.Module): loss layer
        encoder_optimizer (Union[Optimizer, None]): optimizer for encoder's weights (if fine-tuning)
        decoder_optimizer (Optimizer): optimizer to update decoder's weights
        epoch (int): epoch number
    """
    # train mode (dropout and batchnorm is used)
    decoder.train()
    encoder.train()

    # forward prop. + back prop. time
    batch_time = AverageMeter()
    # data loading time
    data_time = AverageMeter()
    # loss (per word decoded)
    losses = AverageMeter()
    # top5 accuracy
    top5accs = AverageMeter()

    scaler = GradScaler(enabled=torch.cuda.is_available())
    start = time.time()
    # Batches
    for i, (imgs, captions, captionslens) in enumerate(train_loader):

        data_time.update(time.time() - start)
        # Move to GPU, if available
        imgs = imgs.to(DEVICE)
        if encoder_optimizer is None:
            imgs_encode = encoder(imgs)
        for cap_index in range(num_captions):
            caps, caplens = captions[cap_index], captionslens[cap_index]
            caps = caps.to(DEVICE)
            caplens = caplens.to(DEVICE)

            # Forward prop.
            if encoder_optimizer is not None:
                imgs_encode = encoder(imgs)
            with autocast(enabled=torch.cuda.is_available()):
                scores, caps_sorted, decode_lens, alphas, _ = decoder(
                    imgs_encode, caps, caplens
                )

                # We decode starting with <SOS>, targets are all words after <SOS> -> <EOS>
                targets = caps_sorted[:, 1:].long()

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
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
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(decoder_optimizer)
            scaler.update()

            # Back prop.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                scaler.step(encoder_optimizer)
                encoder_optimizer.zero_grad()

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
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                    f"Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})"
                )


def validate(
    val_loader: DataLoader, encoder: nn.Module, decoder: nn.Module, criterion: nn.Module
) -> float:
    """
    Performs one epoch's validation.

    Args:
        val_loader (DataLoader): DataLoader for validation data.
        encoder (nn.Module): encoder model
        decoder (nn.Module): decoder model
        criterion (nn.Module): loss layer

    Returns:
        float: BLEU-4 score
    """
    # eval mode (no dropout or batchnorm)
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()
    # references (true captions) for calculating BLEU-4 score
    references = []
    # hypotheses (predictions)
    hypotheses = []

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, captions, captionslens) in enumerate(val_loader):
            # Move to GPU, if available
            imgs = imgs.to(DEVICE)
            # Forward prop.
            if encoder is not None:
                imgs_encode = encoder(imgs)
            reference_caps = []
            for cap_index in range(num_captions):
                caps, caplens = captions[cap_index], captionslens[cap_index]
                caps = caps.to(DEVICE)
                caplens = caplens.to(DEVICE)

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
                        f"Validation: [{i}/{len(val_loader)}]\t"
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
            # for n images, with n hypotheses, and references a, b, c... for each image
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...]
            # hypotheses = [hyp1, hyp2, ...]

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


def infer(num_workers: int = 4) -> None:
    """
    Inference
    """
    vocab, _, _, _ = generators(
        seq_length=25, batch_size=batch_size, num_workers=num_workers
    )
    vocab_size = len(vocab)
    # Initialize / load checkpoint
    global checkpoint

    print(f"Loading trained model from {checkpoint}.")
    checkpoint = torch.load(checkpoint, map_location=torch.device(DEVICE))
    decoder = checkpoint["decoder"]
    encoder = checkpoint["encoder"]

    # Move to GPU, if available
    decoder = decoder.to(DEVICE)
    encoder = encoder.to(DEVICE)

    images = [
        "datasets/Flicker8k_Dataset/1002674143_1b742ab4b8.jpg",
        "datasets/Flicker8k_Dataset/1030985833_b0902ea560.jpg",
    ]
    # Data augmentation for image data
    image_size = (224, 224)
    img_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    for _, img_path in enumerate(images):
        k = 3  # beam_size
        img = Image.open(img_path).convert("RGB")
        # (1, 3, img_size, img_size)
        img = img_transform(img).unsqueeze(0).to(DEVICE)

        # Encode # (1, enc_image_size, enc_image_size, encoder_dim)
        encoder_out = encoder(img)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding # (1, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        # (k, num_pixels, encoder_dim)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <SOS>. # (k, 1)
        k_prev_words = torch.LongTensor([[vocab.stoi["<SOS>"]]] * k).to(DEVICE)

        # Tensor to store top k sequences; now they're just <SOS> # (k, 1)
        seqs = k_prev_words

        # Tensor to store top k sequences' scores; now they're just 0 # (k, 1)
        top_k_scores = torch.zeros(k, 1).to(DEVICE)

        # Lists to store completed sequences and scores
        complete_seqs = []
        complete_seqs_scores = []

        # Start decoding
        step = 1
        hidden, cell = decoder.init_hidden_state(encoder_out)

        # s <= k, because sequences are removed from this process once they hit <EOS>
        while True:
            # (s, embed_dim)
            embeddings = decoder.embedding(k_prev_words).squeeze(1)
            # (s, encoder_dim), (s, num_pixels)
            awe, _ = decoder.attention(encoder_out, hidden)
            # gating scalar, (s, encoder_dim)
            gate = decoder.sigmoid(decoder.f_beta(hidden))
            awe = gate * awe

            # (s, decoder_dim)
            hidden, cell = decoder.decode_step(
                torch.cat([embeddings, awe], dim=1), (hidden, cell)
            )
            # (s, vocab_size)
            scores = decoder.fc(hidden)
            scores = F.log_softmax(scores, dim=1)

            # Add # (s, vocab_size)
            scores = top_k_scores.expand_as(scores) + scores

            # Ist step, all k points will have the same scores
            # (since same k previous words, hidden, cell) # (s)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                # Unroll and find top scores, and their unrolled indices # (s)
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            # (s, step+1)
            seqs = torch.cat(
                [seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1
            )

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [
                ind
                for ind, next_word in enumerate(next_word_inds)
                if next_word != vocab.stoi["<EOS>"]
            ]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            # reduce beam length accordingly
            k -= len(complete_inds)

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            prev_word = prev_word_inds[incomplete_inds].long()
            hidden = hidden[prev_word]
            cell = cell[prev_word]
            encoder_out = encoder_out[prev_word]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        seq = [vocab.itos[token_index] for token_index in seq[1:-1]]
        print(seq)


if __name__ == "__main__":
    # main()
    infer()
