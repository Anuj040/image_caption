"""utilities for data processing"""
import os
import zipfile
from typing import Tuple

import numpy as np
import pandas as pd

np.random.seed(123)


def load_captions_data(
    filepath: str, images_path: str, min_seq_length: int = 5, max_seq_length: int = 25
):
    """Loads captions (text) data and maps them to corresponding images.
    https://keras.io/examples/vision/image_captioning/#preparing-the-dataset

    Args:
        filepath (str): Path to the text file containing caption data.
        images_path (str):
        min/max_seq_length (int): length allowed for any sequence

    Returns:
        caption_mapping: Dictionary mapping image names and the corresponding captions
        text_data: List containing all the available captions
    """

    with open(filepath, "r", encoding="utf-8") as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = {}
        text_data = []
        images_to_skip = set()

        for line in caption_data:
            line = line.rstrip("\n")
            # Image name and captions are separated using a tab
            img_name, caption = line.split("\t")

            # Each image is repeated five times for the five different captions.
            # Each image name has a suffix `#(caption_number)`
            img_name = img_name.split("#")[0]
            img_name = os.path.join(images_path, img_name.strip())

            # We will remove caption that are either too short to too long
            tokens = caption.strip().split()

            if len(tokens) < min_seq_length or len(tokens) > max_seq_length:
                images_to_skip.add(img_name)
                continue

            if img_name.endswith("jpg") and img_name not in images_to_skip:
                # Gather all the text data
                text_data.append(caption)

                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]

        for img_name in images_to_skip:
            if img_name in caption_mapping:
                del caption_mapping[img_name]

        return caption_mapping, text_data


def train_val_split(
    caption_data: dict, train_size: float = 0.8, shuffle: bool = True
) -> Tuple[dict]:
    """Split the captioning dataset into train and validation sets.

    Args:
        caption_data (dict): Dictionary containing the mapped caption data
        train_size (float): Fraction of all the full dataset to use as training data
        shuffle (bool): Whether to shuffle the dataset before splitting

    Returns:
        Traning and validation datasets as two separated dicts
    """

    # 1. Get the list of all image names
    all_images = list(caption_data.keys())

    # 2. Shuffle if necessary
    if shuffle:
        np.random.shuffle(all_images)

    # 3. Split into training and validation sets
    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    # 4. Return the splits
    return training_data, validation_data


def data_downloader(path: str) -> None:
    """Method for downloading the dataset if not available at the directed location
    Args:
        path (str): Directed location
    """
    os.makedirs(path, exist_ok=True)
    url = "http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"
    os.system(f"wget {url} -P {path}")

    glove_path = os.path.join(path, "glove.6B.zip")

    # Extrcting the contents of the downloaded file
    with zipfile.ZipFile(glove_path, "r") as zip_ref:
        zip_ref.extractall(path)

    # Cleaning # Remove .zip file
    os.remove(glove_path)


def prepare_embeddings(path: str, vocab: dict, embed_dim: int = 100) -> np.ndarray:
    """prepares pretrained token embeddings matrix

    Args:
        path (str): path to the embeddings file
        vocab (dict): vocabulary object
        embed_dim (int, optional): embedding dimensions for each word. Defaults to 100.

    Returns:
        np.ndarray: embedding matrix with row number corresponing to word index in vocab
    """
    glove_path = os.path.join(path, f"glove.6B.{embed_dim}d.txt")
    # Check if the specified embeddings available
    if not os.path.exists(glove_path):
        data_downloader(path)

    glove = pd.read_csv(glove_path, sep=" ", quoting=3, header=None, index_col=0)
    glove_embedding = {key: val.values for key, val in glove.T.items()}

    embedding_matrix = np.zeros((len(vocab) + 1, embed_dim))
    not_in_glove_index = []
    for word, index in vocab.stoi.items():
        if word in glove_embedding:
            embedding_matrix[index] = glove_embedding[word]
        else:
            not_in_glove_index.append(index)

    embed_dim_ext = len(not_in_glove_index)
    expanded_embedding_matrix = np.zeros((len(vocab) + 1, embed_dim + embed_dim_ext))
    expanded_embedding_matrix[..., :embed_dim] = embedding_matrix
    placeholder = embed_dim
    for index in not_in_glove_index:
        expanded_embedding_matrix[index, placeholder] = 1.0
        placeholder += 1

    return expanded_embedding_matrix, embed_dim + embed_dim_ext
