import re

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

    cleaned = [line.lower() for line in cleaned]

    return cleaned

def create_pairs(text: list[str], window_size: int) -> list[tuple[str, str]]:
    """
    Create skip-gram (center, context) word pairs.
    Assumes `text` is a list of sentences.
    """

    pairs = []

    for line in text:
        words = line.split()

        for idx in range(len(words)):
            center_word = words[idx]

            lower = max(0, idx - window_size)
            upper = min(len(words), idx + window_size + 1)

            for ctx_idx in range(lower, upper):
                if ctx_idx == idx:
                    continue

                context_word = words[ctx_idx]
                pairs.append((center_word, context_word))

    return pairs

def create_vocabulary(pairs: list[tuple[str, str]], threshold: int = 0) -> list[str]:
    vocabulary = {}

    for pair in pairs:
        if pair[0] not in vocabulary:
            vocabulary[pair[0]] = 1
        else:
            vocabulary[pair[0]] += 1

    subset = {word for word, amount in vocabulary.items() if amount > threshold}

    return list(subset)

def filter_pairs_from_vocabulary(pairs: list[tuple[str, str]], vocabulary: list[str]) -> list[tuple[str, str]]:
    filtered_pairs = []

    for pair in pairs:
        if pair[0] in vocabulary and pair[1] in vocabulary: filtered_pairs.append(pair)
    
    return filtered_pairs

def encode_pairs(pairs: list[tuple[str, str]], vocabulary: list[str]) -> list[tuple[int, int]]:
    encoded_pairs = []

    for pair in pairs:
        encoded_pairs.append((vocabulary.index(pair[0]), vocabulary.index(pair[1])))

    return encoded_pairs
