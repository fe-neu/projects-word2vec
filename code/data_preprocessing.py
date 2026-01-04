import re
from typing import Iterator

def load_data() -> list[str]:
    '''
    Loading simple wikipedia articles as list of lines.

    Precleaned and lower case.
    
    :return: Cleaned Text
    :rtype: list[str]
    '''

    text = []

    with open("../data/text.txt", encoding="utf-8") as f:
        text = f.readlines()

    text = [line for line in text if line != "\n"]

    cleaned = [
        re.sub(r"\s+", " ",
            re.sub(r"[^a-zA-Z ]+", "", line)
        ).strip()
        for line in text
    ]

    cleaned = [line.lower().split() for line in cleaned]

    return cleaned

def generate_pairs(text: list[list[int | None]], window_radius: int) -> Iterator[tuple[int, int]]:
    """
    Create skip-gram (center, context) word pairs.
    Assumes `text` is a list of sentences.
    """

    for line in text:
        for idx in range(len(line)):
            center_word = line[idx]
            if center_word is None: continue

            lower = max(0, idx - window_radius)
            upper = min(len(line), idx + window_radius + 1)

            for ctx_idx in range(lower, upper):
                if ctx_idx == idx:
                    continue

                context_word = line[ctx_idx]

                if context_word is not None:
                    yield center_word, context_word

def encode_word(word: str, vocabulary_map: dict[str, int]) -> int | None:
    return vocabulary_map.get(word)

def encode_text(text: list[list[str]], vocabulary_map: dict[str, int]) -> list[list[int]]:
    encoded_text = []
    for line in text:
        encoded_line = []
        for word in line:
            encoded_line.append(encode_word(word, vocabulary_map))
        encoded_text.append(encoded_line)

    return encoded_text

def create_vocabulary(text: list[list[str]], threshold: int = 0) -> list[str]:
    vocabulary = {}

    for line in text:
        for word in line:
            if word not in vocabulary:
                vocabulary[word] = 1
            else:
                vocabulary[word] += 1

    subset = {word for word, amount in vocabulary.items() if amount > threshold}

    return list(subset)

def create_vocabulary_map(vocabulary: list[str]) -> set[str, int]:
    word2id = {word: i for i, word in enumerate(vocabulary)}

    return word2id