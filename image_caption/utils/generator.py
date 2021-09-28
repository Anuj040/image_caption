"""data generator module"""
import os
import random
import re
from typing import Callable, List

import spacy
import torch
from PIL import Image

SPACY_ENG = spacy.load("en_core_web_sm")
# pylint: disable = wrong-import-position
from image_caption.utils.data_utils import load_captions_data, train_val_split

random.seed(111)


class Vocabulary:
    """Vocabulary building object"""

    def __init__(self, freq_threshold: int, standardize: Callable):
        """Initializer

        Args:
            freq_threshold (int): Neglect words with occurence less than the threshold
        """
        self.standardize = standardize

        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.default_dict_len = len(self.stoi)

        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
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
                else:
                    frequency[word] += 1

                if frequency[word] > self.freq_threshold - 1:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def numericalize(self, sentence: str) -> List[int]:
        """returns a vector of integers representing individual word in a phrase

        Args:
            sentence (str): input string

        Returns:
            List[int]: vector representation of the string
        """
        tokenized_text = self.tokenizer_eng(sentence)

        return [
            self.stoi[word] if word in self.stoi else self.stoi["<UNK>"]
            for word in tokenized_text
        ]


# pylint: disable = too-many-arguments
class CaptionDataset:
    """Prepares the flicker image caption dataset (base)"""

    def __init__(
        self,
        root_dir: str = "datasets",
        caption_file: str = "Flickr8k.token.txt",
        freq_threshold: int = 5,
        transform=None,
        seq_length: int = 25,
        split: str = "train",
    ) -> None:
        """Initializes

        Args:
            root_dir (str, optional): Defaults to "datasets".
            caption_file (str, optional): name of the captions file.
                    Defaults to "Flickr8k.token.txt".
            freq_threshold (int, optional): Neglect words with occurence less than the threshold
            transform ([type], optional): Image transformations Defaults to None.
            seq_length (int, optional): max caption length for the dataset prep. Defaults to 25.
            split (str): data split to return
        """
        self.freq_threshold = freq_threshold
        self.transform = transform
        self.root_dir = root_dir

        caption_path = os.path.join(root_dir, caption_file)
        images_path = os.path.join(root_dir, "Flicker8k_Dataset")

        # Load the dataset
        captions_mapping, text_data = load_captions_data(
            caption_path, images_path, max_seq_length=seq_length
        )
        train_data, valid_data = train_val_split(captions_mapping)
        self.captions = train_data if split == "train" else valid_data

        self.images = list(self.captions.keys())

        # strip specific characters from the string
        strip_chars = r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        strip_chars = strip_chars.replace("<", "")
        self.strip_chars = strip_chars.replace(">", "")

        self.vocab = Vocabulary(freq_threshold, self.custom_standardization)
        self.vocab.build_vocabulary(text_data)

        # # Fixed length allowed for any sequence
        # self.seq_len = seq_length

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        #! TODO: check if the model can work with all captions
        caption = random.choice(self.captions[image])

        img = Image.open(image).convert("RGB")

        if self.transform:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.Tensor(numericalized_caption)

    def custom_standardization(self, input_string):
        """custom function for removing certain specific substrings from the phrase"""
        return re.sub(f"[{re.escape(self.strip_chars)}]", "", input_string)


if __name__ == "__main__":  # pragma: no cover
    dataset = CaptionDataset()

    out = dataset[100]
    import numpy as np

    out[0].show()
    print([dataset.vocab.itos[key] for key in np.array(out[1])])
