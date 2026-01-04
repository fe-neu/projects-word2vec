import numpy as np

def create_frequency_array(text: list[list[int | None]], vocabulary_size: int):
    frequencies = np.zeros(vocabulary_size, dtype=np.int64)

    for line in text:
        for word in line:
            if word is not None:
                frequencies[word] += 1

    return frequencies

def get_neg_sampling_dist(frequencies: np.ndarray) -> np.ndarray:
    neg_sampling_dist = frequencies ** 0.75
    neg_sampling_dist /= neg_sampling_dist.sum()
    return neg_sampling_dist

def sample_negative_contexts(
    k: int,
    neg_sampling_dist: np.ndarray,
    forbidden: int | None = None
) -> list[int]:
    """
    Sample k negative word IDs.
    Optionally avoid sampling the true context word.
    """

    negatives = []

    while len(negatives) < k:
        sampled = np.random.choice(
            len(neg_sampling_dist),
            p=neg_sampling_dist
        )
        if sampled != forbidden:
            negatives.append(sampled)

    return negatives