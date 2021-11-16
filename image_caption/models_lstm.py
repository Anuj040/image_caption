"""https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py"""
from typing import Tuple

import torch
import torchvision
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size: int = 14):
        super().__init__()
        self.enc_image_size = encoded_image_size

        # pretrained ImageNet ResNet-101
        resnet = torchvision.models.resnet101(pretrained=True)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

        self.fine_tune()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            images (torch.Tensor): images, (batch_size, 3, image_size, image_size)

        Returns:
            torch.Tensor: encoded images
        """
        # (batch_size, 2048, image_size/32, image_size/32)
        out = self.resnet(images)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune: bool = True) -> None:
        """
        Tune the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        Args:
            fine_tune (bool, optional): Allow?. Defaults to True.
        """
        for param in self.resnet.parameters():
            param.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for child in list(self.resnet.children())[5:]:
            for param in child.parameters():
                param.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super().__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        # softmax layer to calculate weights
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """[summary]

        Args:
            encoder_out (torch.Tensor): encoded images, a tensor of dimension
                                    (batch_size, num_pixels, encoder_dim)
            decoder_hidden (torch.Tensor): previous decoder output
                                    (batch_size, decoder_dim)

        Returns:
            torch.Tensor: attention weighted encoding
            torch.Tensor: weights
        """

        # (batch_size, num_pixels, attention_dim)
        att1 = self.encoder_att(encoder_out)
        # (batch_size, attention_dim)
        att2 = self.decoder_att(decoder_hidden)
        # (batch_size, num_pixels)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        # (batch_size, num_pixels)
        alpha = self.softmax(att)
        # (batch_size, encoder_dim)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(
        self,
        attention_dim: int,
        embed_dim: int,
        decoder_dim: int,
        vocab_size: int,
        encoder_dim: int = 2048,
        dropout: float = 0.5,
    ):
        """[summary]

        Args:
            attention_dim (int): size of attention network
            embed_dim (int): embedding size
            decoder_dim (int): size of decoder's RNN
            vocab_size (int): size of vocabulary
            encoder_dim (int, optional): feature size of encoded images. Defaults to 2048.
            dropout (float, optional): dropout. Defaults to 0.5.
        """
        super().__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # attention network
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        # decoding LSTMCell
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        # linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        # linear layer to find scores over vocabulary
        self.fc = nn.Linear(decoder_dim, vocab_size)
        # initialize some layers with the uniform distribution
        self.init_weights()

    def init_weights(self) -> None:
        """
        Initializes some parameters with uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune: bool = True) -> None:
        """
        Allow fine-tuning of embedding layer? (not-allow if using pre-trained embeddings).

        Args:
            fine_tune (bool, optional): Allow? Defaults to True.
        """
        for params in self.embedding.parameters():
            params.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on encoded images.

        Args:
            encoder_out (torch.Tensor): encoded images, a tensor of dimension
                            (batch_size, num_pixels, encoder_dim)

        Returns:
            Tuple[torch.Tensor]: hidden state, cell state
        """

        mean_encoder_out = encoder_out.mean(dim=1)
        # (batch_size, decoder_dim)
        hidden = self.init_h(mean_encoder_out)
        cell = self.init_c(mean_encoder_out)
        return hidden, cell

    def forward(self, encoder_out: torch.Tensor, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image # (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(
            dim=0, descending=True
        )
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding # (batch_size, max_caption_length, embed_dim)
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state # (batch_size, decoder_dim)
        hidden, cell = self.init_hidden_state(encoder_out)

        # We decode starting with <SOS>, targets are all words after <SOS> -> <EOS>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(
            DEVICE
        )
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(DEVICE)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum(l > t for l in decode_lengths)
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], hidden[:batch_size_t]
            )
            # gating scalar, (batch_size_t, encoder_dim)
            gate = self.sigmoid(self.f_beta(hidden[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            # (batch_size_t, decoder_dim)
            hidden, cell = self.decode_step(
                torch.cat(
                    [embeddings[:batch_size_t, t, :], attention_weighted_encoding],
                    dim=1,
                ),
                (hidden[:batch_size_t], cell[:batch_size_t]),
            )
            # (batch_size_t, vocab_size)
            preds = self.fc(self.dropout(hidden))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
