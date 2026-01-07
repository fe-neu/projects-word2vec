import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid activation function in a numerically stable way.

    The input values are clipped to a fixed range to avoid numerical overflow
    when computing the exponential.

    :param x: Input value or array.
    :type x: float | np.ndarray
    :return: Sigmoid-transformed value(s).
    :rtype: float | np.ndarray
    """
    x = np.clip(x, -15, 15)
    return 1.0 / (1.0 + np.exp(-x))


class Embedder:
    """
    Word embedding model implementing the skip-gram architecture with
    negative sampling.

    The class maintains separate input and output embedding matrices and
    provides forward and backward passes for training word embeddings.
    """

    def __init__(
        self,
        embed_dims: int,
        vocabulary_size: int,
        random_seed: int = 42
    ):
        """
        Initialize the embedding matrices.

        Input and output embeddings are initialized with small random values
        drawn from a normal distribution.

        :param embed_dims: Dimensionality of the embedding vectors.
        :type embed_dims: int
        :param vocabulary_size: Number of unique tokens in the vocabulary.
        :type vocabulary_size: int
        :param random_seed: Seed for the random number generator to ensure
                            reproducibility.
        :type random_seed: int
        """
        self.embed_dims = embed_dims
        self.vocabulary_size = vocabulary_size

        rng = np.random.default_rng(random_seed)

        self.input_weights = rng.normal(
            scale=0.01,
            size=(vocabulary_size, embed_dims)
        )
        self.output_weights = rng.normal(
            scale=0.01,
            size=(vocabulary_size, embed_dims)
        )

    def forward(
        self,
        center_word: int,
        true_context: int,
        negative_samples: list[int]
    ):
        """
        Perform the forward pass for a single skip-gram training example.

        This method retrieves the embedding vectors corresponding to the
        center word, the true context word, and the negative sample words.

        :param center_word: Index of the center (input) word.
        :type center_word: int
        :param true_context: Index of the true context (output) word.
        :type true_context: int
        :param negative_samples: Indices of negatively sampled words.
        :type negative_samples: list[int]
        :return: Tuple containing the center word vector, the true context
                 vector, and the negative sample vectors.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        v_c = self.input_weights[center_word]
        u_o = self.output_weights[true_context]
        u_ni = self.output_weights[negative_samples]

        return v_c, u_o, u_ni

    def backward(
        self,
        learning_rate: float,
        v_c: np.ndarray,
        u_o: np.ndarray,
        u_ni: np.ndarray,
        center_word: int,
        true_context: int,
        negative_samples: list[int]
    ) -> float:
        """
        Perform the backward pass and update embedding weights.

        Gradients are computed for the center word, the positive context word,
        and the negative samples according to the negative sampling objective.
        The embedding matrices are updated in-place.

        :param learning_rate: Learning rate used for the parameter updates.
        :type learning_rate: float
        :param v_c: Embedding vector of the center word.
        :type v_c: np.ndarray
        :param u_o: Embedding vector of the true context word.
        :type u_o: np.ndarray
        :param u_ni: Embedding vectors of the negative sample words.
        :type u_ni: np.ndarray
        :param center_word: Index of the center (input) word.
        :type center_word: int
        :param true_context: Index of the true context (output) word.
        :type true_context: int
        :param negative_samples: Indices of negatively sampled words.
        :type negative_samples: list[int]
        :return: Scalar loss value for the current training example.
        :rtype: float
        """
        x_o = v_c @ u_o
        x_ni = u_ni @ v_c

        sig_o = sigmoid(x_o)
        sig_ni = sigmoid(x_ni)

        grad_center = (
            (sig_o - 1.0) * u_o
            + np.sum(sig_ni[:, None] * u_ni, axis=0)
        )

        grad_pos = (sig_o - 1.0) * v_c
        grad_neg = sig_ni[:, None] * v_c

        self.input_weights[center_word] -= learning_rate * grad_center
        self.output_weights[true_context] -= learning_rate * grad_pos
        self.output_weights[negative_samples] -= learning_rate * grad_neg

        loss = (
            -np.log(sig_o)
            -np.sum(np.log(sigmoid(-x_ni)))
        )

        return loss
