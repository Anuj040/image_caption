"""
data generator module
https://www.kaggle.com/fanbyprinciple/pytorch-image-captioning-with-flickr/notebook
"""
import os
import random
import re
from typing import Callable, List, Tuple

import numpy as np
import spacy
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

SPACY_ENG = spacy.load("en_core_web_sm")
# pylint: disable = wrong-import-position
from image_caption.utils.data_utils import load_captions_data, train_val_split


# pylint: disable = attribute-defined-outside-init
class Vocabulary:
    """Vocabulary building object"""

    def __init__(self, standardize: Callable):
        """Initializer

        Args:
            standardize (Callable): utility function for standardizing the text inputs
        """

        self.standardize = standardize

        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text: str):
        """creates a token vector from a literal phrase"""
        text = self.standardize(text)
        return [tok.text.lower() for tok in SPACY_ENG.tokenizer(text)]

    def build_vocabulary(self, sentences: List[str]) -> None:
        """Builds integer key-word and vice-versa dictionaries

        Args:
            sentences (List[str]): list of phrases
        """
        idx = 4
        frequency = {}

        for sentence in sentences:
            for word in self.tokenizer_eng(sentence):
                if word not in frequency:
                    frequency[word] = 1
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1
                else:
                    frequency[word] += 1
        self.weights = np.ones(len(self.itos)) * len(sentences)
        for idx in range(4, len(self.itos)):
            word = self.itos[idx]
            freq = frequency[word]
            self.weights[idx] = freq
        self.weights = 1 / self.weights ** (0.4)
        self.weights = self.weights / min(self.weights)
        # self.weights = (len(sentences) - self.weights) / self.weights
        self.weights = np.expand_dims(np.expand_dims(self.weights, axis=0), axis=-1)

    def numericalize(self, sentence: str) -> List[int]:
        """returns a vector of integers representing individual word in a phrase

        Args:
            sentence (str): input string

        Returns:
            List[int]: vector representation of the string
        """
        tokenized_text = self.tokenizer_eng(sentence)

        return (
            [self.stoi["<SOS>"]]
            + [
                self.stoi[word] if word in self.stoi else self.stoi["<UNK>"]
                for word in tokenized_text
            ]
            + [self.stoi["<EOS>"]]
        )


# pylint: disable = too-many-arguments
class CaptionDataset(Dataset):
    """Prepares the flicker image caption dataset (base)"""

    def __init__(
        self,
        root_dir: str = "datasets",
        caption_file: str = "Flickr8k.token.txt",
        transform=None,
        seq_length: int = 25,
        split: str = "train",
        with_mask: bool = False,
    ) -> None:
        """Initializes

        Args:
            root_dir (str, optional): Defaults to "datasets".
            caption_file (str, optional): name of the captions file.
                    Defaults to "Flickr8k.token.txt".
            transform ([type], optional): Image transformations Defaults to None.
            seq_length (int, optional): max caption length for the dataset prep. Defaults to 25.
            split (str): data split to return
        """
        self.transform = transform
        self.root_dir = root_dir
        self.with_mask = with_mask

        caption_path = os.path.join(root_dir, caption_file)
        images_path = os.path.join(root_dir, "Flicker8k_Dataset")

        # Load the dataset
        captions_mapping, text_data = load_captions_data(
            caption_path, images_path, max_seq_length=seq_length
        )
        # strip specific characters from the string
        strip_chars = r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        strip_chars = strip_chars.replace("<", "")
        self.strip_chars = strip_chars.replace(">", "")
        # Build the vocab
        self.vocab = Vocabulary(self.custom_standardization)
        self.vocab.build_vocabulary(text_data)

        if with_mask:
            # Add mask element # Update stoi before itos
            self.vocab.stoi["<MASK>"] = len(self.vocab)
            self.vocab.itos[len(self.vocab)] = "<MASK>"

        # Prepare the split # Returns vectorized captions
        train_data, valid_data = train_val_split(
            captions_mapping, self.vocab.numericalize
        )
        self.captions = train_data if split == "train" else valid_data

        self.images = list(self.captions.keys())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        img = Image.open(image).convert("RGB")

        if self.transform:
            img = self.transform(img)

        numericalized_captions = self.captions[image]
        caption_lens = [[len(caption)] for caption in numericalized_captions]

        if self.with_mask:
            masked_captions = []
            output_labels = []
            # Mask some of the tokens
            for caption in numericalized_captions:
                inputs, outputs = self.random_word(caption)
                masked_captions.append(inputs)
                output_labels.append(outputs)
        else:
            masked_captions = output_labels = numericalized_captions

        return img, masked_captions, caption_lens, output_labels

    def custom_standardization(self, input_string):
        """custom function for removing certain specific substrings from the phrase"""
        return re.sub(f"[{re.escape(self.strip_chars)}]", "", input_string)

    def random_word(self, sequence: torch.Tensor) -> Tuple[torch.Tensor]:
        """Prepares a partially masked input sequence"""
        output_label = sequence.clone()

        # Skip the <SOS> token
        for i, _ in enumerate(sequence[1:], start=1):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    sequence[i] = self.vocab.stoi["<MASK>"]

                # 10% randomly change token to random token
                elif prob < 0.9:
                    sequence[i] = random.randrange(len(self.vocab))

                # 10% randomly leave the token unchanged

        return sequence, output_label


class Collate:
    """process the list of samples to form a batch"""

    def __init__(self, pad_value: int, num_captions: int = 5):
        """intialize

        Args:
            pad_value (int): value to pad the sequence with
            num_captions (int): number of captions for each image
        """
        self.pad_value = pad_value
        self.num_captions = num_captions

    def __call__(self, batch: list) -> Tuple[torch.Tensor]:
        """returns the batch from input lists"""
        imgs = [item[0].unsqueeze(0) for item in batch]
        img = torch.cat(imgs, dim=0)

        captions_inputs = []
        captions_targets = []
        captions_lens = []
        for i in range(self.num_captions):
            inputs = [item[1][i] for item in batch]
            inputs = pad_sequence(
                inputs, batch_first=True, padding_value=self.pad_value
            )
            captions_inputs.append(inputs)

            lengths = [item[2][i] for item in batch]
            captions_lens.append(lengths)

            targets = [item[3][i] for item in batch]
            targets = pad_sequence(
                targets, batch_first=True, padding_value=self.pad_value
            )
            captions_targets.append(targets)

        return (
            img,
            captions_inputs,
            captions_targets,
            torch.Tensor(captions_lens).to(dtype=torch.int32),
        )


if __name__ == "__main__":  # pragma: no cover
    dataset = CaptionDataset()

    out = dataset[100]

    out[0].show()
    print([dataset.vocab.itos[key] for key in np.array(out[1])])
