import numpy as np


def sigmoid(x):
    # Numerically stable sigmoid
    x = np.clip(x, -15, 15)
    return 1.0 / (1.0 + np.exp(-x))


class Embedder:

    def __init__(
        self,
        embed_dims: int,
        vocabulary_size: int,
        random_seed: int = 42
    ):
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
        v_c = self.input_weights[center_word]              # (d,)
        u_o = self.output_weights[true_context]            # (d,)
        u_ni = self.output_weights[negative_samples]       # (k, d)

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

        # Scores
        x_o = v_c @ u_o            # scalar
        x_ni = u_ni @ v_c          # (k,)

        # Sigmoids
        sig_o = sigmoid(x_o)
        sig_ni = sigmoid(x_ni)

        # Gradients
        grad_center = (
            (sig_o - 1.0) * u_o
            + np.sum(sig_ni[:, None] * u_ni, axis=0)
        )

        grad_pos = (sig_o - 1.0) * v_c
        grad_neg = sig_ni[:, None] * v_c

        # Updates
        self.input_weights[center_word] -= learning_rate * grad_center
        self.output_weights[true_context] -= learning_rate * grad_pos
        self.output_weights[negative_samples] -= learning_rate * grad_neg

        loss = (
            -np.log(sig_o)
            -np.sum(np.log(sigmoid(-x_ni)))
        )

        return loss

