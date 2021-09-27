"""data generator module"""
import os
import re

import spacy
from image_caption.utils.data_utils import load_captions_data

SPACY_ENG = spacy.load("en_core_web_sm")


class Vocabulary:
    """Vocabulary building object"""

    def __init__(self, freq_threshold: int):
        """Initializer

        Args:
            freq_threshold (int): Neglect words with occurence less than the threshold
        """

        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in SPACY_ENG.tokenizer(text)]

    def build_vocabulary(self, sentences):
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

    def numericalize(self, sentence):
        tokenized_text = self.tokenizer_eng(sentence)

        return [
            self.stoi[word] if word in self.stoi else self.stoi["<UNK>"]
            for word in tokenized_text
        ]


class CaptionDataset:
    def __init__(
        self,
        root_dir="datasets",
        caption_file="Flickr8k.token.txt",
        freq_threshold=5,
        transform=None,
        vocab_size: int = 10000,
        seq_length: int = 25,
    ) -> None:
        self.freq_threshold = freq_threshold
        self.transform = transform
        self.root_dir = root_dir

        caption_path = os.path.join(root_dir, caption_file)
        images_path = os.path.join(root_dir, "Flicker8k_Dataset")

        # Load the dataset
        captions_mapping, text_data = load_captions_data(
            caption_path, images_path, max_seq_length=seq_length
        )

        self.captions = captions_mapping.values()
        self.images = captions_mapping.keys()

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(text_data)

        strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        strip_chars = strip_chars.replace("<", "")
        self.strip_chars = strip_chars.replace(">", "")

        # # Vocabulary size
        # self.vocab_size = vocab_size

        # # Fixed length allowed for any sequence
        # self.seq_len = seq_length

    def custom_standardization(self, input_string):
        return re.sub("[%s]" % re.escape(self.strip_chars), "", input_string)


if __name__ == "__main__":
    dataset = CaptionDataset()

    out = dataset.custom_standardization(
        "A toddler in blue shorts is laying face down on the wet ground ."
    )
    print(out)
