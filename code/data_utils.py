"""
Utility functions for loading, cleaning, encoding, and preprocessing text data
for word embedding models such as skip-gram word2vec.

The module provides helpers to:
- Download and load the Simple English Wikipedia dataset
- Clean and normalize raw text
- Build vocabularies and word-to-id mappings
- Encode text into integer representations
- Generate skip-gram (center, context) training pairs
"""

import re
from typing import Iterator


def load_simple_wiki_dataset() -> list[str]:
    """
    Load the Simple English Wikipedia text dataset.

    The dataset is downloaded from Kaggle:
    https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish/

    If the dataset is not already cached locally, it will be downloaded
    automatically using ``kagglehub``.

    :return: A list of lines from the Simple Wikipedia dataset.
    :rtype: list[str]
    """
    import kagglehub

    path = kagglehub.dataset_download("ffatty/plain-text-wikipedia-simpleenglish")

    data = []

    with open(path + "/AllCombined.txt", encoding="utf-8") as f:
        data = f.readlines()

    return data


def preclean_text(text: list[str]) -> list[str]:
    """
    Remove empty lines from a list of text lines.

    Lines consisting only of a newline character (``"\\n"``) are filtered out.

    :param text: List of raw text lines.
    :type text: list[str]
    :return: List of text lines without empty newline-only entries.
    :rtype: list[str]
    """
    return [line for line in text if line != "\n"]


def clean_text(text: list[str]) -> list[list[str]]:
    """
    Normalize and tokenize text.

    The cleaning process:
    - Removes all non-letter characters (keeps only a–z and A–Z)
    - Collapses multiple whitespace characters into a single space
    - Converts all text to lowercase
    - Splits each line into a list of tokens

    :param text: List of raw text lines.
    :type text: list[str]
    :return: Tokenized and cleaned text, where each line is a list of words.
    :rtype: list[list[str]]
    """
    cleaned = [
        re.sub(
            r"\s+",
            " ",
            re.sub(r"[^a-zA-Z ]+", "", line),
        ).strip()
        for line in text
    ]

    cleaned = [line.lower().split() for line in cleaned]

    return cleaned


def generate_pairs(
    text: list[list[int | None]],
    window_radius: int
) -> Iterator[tuple[int, int]]:
    """
    Generate skip-gram (center, context) word pairs.

    For each word in each sentence, all neighboring words within the given
    window radius are yielded as context words. Entries with value ``None``
    are skipped and never emitted as part of a pair.

    :param text: Encoded text represented as a list of sentences, where each
                 sentence is a list of word indices or ``None``.
    :type text: list[list[int | None]]
    :param window_radius: Number of tokens to consider on each side of the
                          center word.
    :type window_radius: int
    :yield: Tuples of the form ``(center_word_id, context_word_id)``.
    :rtype: Iterator[tuple[int, int]]
    """
    for line in text:
        for idx in range(len(line)):
            center_word = line[idx]
            if center_word is None:
                continue

            lower = max(0, idx - window_radius)
            upper = min(len(line), idx + window_radius + 1)

            for ctx_idx in range(lower, upper):
                if ctx_idx == idx:
                    continue

                context_word = line[ctx_idx]

                if context_word is not None:
                    yield center_word, context_word


def encode_word(word: str, vocabulary_map: dict[str, int]) -> int | None:
    """
    Encode a single word as an integer ID.

    If the word is not present in the vocabulary map, ``None`` is returned.

    :param word: The word to encode.
    :type word: str
    :param vocabulary_map: Mapping from words to integer IDs.
    :type vocabulary_map: dict[str, int]
    :return: Integer ID of the word, or ``None`` if the word is unknown.
    :rtype: int | None
    """
    return vocabulary_map.get(word)


def encode_text(
    text: list[list[str]],
    vocabulary_map: dict[str, int]
) -> list[list[int | None]]:
    """
    Encode tokenized text using a vocabulary mapping.

    Each word in the input text is replaced by its corresponding integer ID.
    Words not present in the vocabulary map are encoded as ``None``.

    :param text: Tokenized text, where each line is a list of words.
    :type text: list[list[str]]
    :param vocabulary_map: Mapping from words to integer IDs.
    :type vocabulary_map: dict[str, int]
    :return: Encoded text as lists of integer IDs or ``None``.
    :rtype: list[list[int | None]]
    """
    encoded_text = []
    for line in text:
        encoded_line = []
        for word in line:
            encoded_line.append(encode_word(word, vocabulary_map))
        encoded_text.append(encoded_line)

    return encoded_text


def create_vocabulary(text: list[list[str]], threshold: int = 0) -> list[str]:
    """
    Create a vocabulary from tokenized text.

    Word frequencies are counted across the entire corpus. Only words with a
    frequency strictly greater than the given threshold are included in the
    resulting vocabulary.

    :param text: Tokenized text, where each line is a list of words.
    :type text: list[list[str]]
    :param threshold: Minimum frequency a word must exceed to be included.
    :type threshold: int
    :return: List of vocabulary words.
    :rtype: list[str]
    """
    vocabulary = {}

    for line in text:
        for word in line:
            if word not in vocabulary:
                vocabulary[word] = 1
            else:
                vocabulary[word] += 1

    subset = {word for word, amount in vocabulary.items() if amount > threshold}

    return list(subset)


def create_vocabulary_map(vocabulary: list[str]) -> dict[str, int]:
    """
    Create a word-to-index mapping from a vocabulary list.

    Each word is assigned a unique integer ID based on its position in the
    vocabulary list.

    :param vocabulary: List of unique vocabulary words.
    :type vocabulary: list[str]
    :return: Dictionary mapping words to integer IDs.
    :rtype: dict[str, int]
    """
    word2id = {word: i for i, word in enumerate(vocabulary)}

    return word2id
