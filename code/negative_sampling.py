import numpy as np


def create_frequency_array(
    text: list[list[int | None]],
    vocabulary_size: int
) -> np.ndarray:
    """
    Create a frequency array for word occurrences in encoded text.

    Iterates over the encoded corpus and counts how often each vocabulary
    index appears. Entries with value ``None`` are ignored.

    :param text: Encoded text represented as a list of sentences, where each
                 sentence is a list of word indices or ``None``.
    :type text: list[list[int | None]]
    :param vocabulary_size: Total number of unique tokens in the vocabulary.
    :type vocabulary_size: int
    :return: Array of word frequencies indexed by vocabulary ID.
    :rtype: np.ndarray
    """
    frequencies = np.zeros(vocabulary_size, dtype=np.int64)

    for line in text:
        for word in line:
            if word is not None:
                frequencies[word] += 1

    return frequencies


def get_neg_sampling_dist(frequencies: np.ndarray) -> np.ndarray:
    """
    Compute the negative sampling distribution from word frequencies.

    The distribution follows the standard word2vec heuristic by raising
    frequencies to the power of 0.75 and normalizing the result to form a
    probability distribution.

    :param frequencies: Array of word frequencies.
    :type frequencies: np.ndarray
    :return: Normalized negative sampling probability distribution.
    :rtype: np.ndarray
    """
    neg_sampling_dist = frequencies ** 0.75
    neg_sampling_dist /= neg_sampling_dist.sum()
    return neg_sampling_dist


def sample_negative_contexts(
    k: int,
    neg_sampling_dist: np.ndarray,
    forbidden: int | None = None
) -> list[int]:
    """
    Sample negative context word indices according to a given distribution.

    Word indices are sampled with replacement based on the provided negative
    sampling distribution. If a forbidden index is specified, it will not be
    included in the sampled negatives.

    :param k: Number of negative samples to draw.
    :type k: int
    :param neg_sampling_dist: Probability distribution over the vocabulary.
    :type neg_sampling_dist: np.ndarray
    :param forbidden: Optional word index that must not be sampled.
    :type forbidden: int | None
    :return: List of sampled negative word indices.
    :rtype: list[int]
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
