import numpy as np

from embedder import Embedder, sigmoid
from data_preprocessing import generate_pairs
from negative_sampling import *
from run import Run

class Trainer:
    def __init__(self,
                 embedder: Embedder,
                 run: Run,
                 text: list[list[int | None]],
                 window_radius: int,
                 vocabulary_size: int,
                 negative_sample_size: int
                 ):
        
        self.embedder = embedder
        self.run = run
        self.text = text
        self.window_radius = window_radius
        self.negative_sample_size = negative_sample_size

        self.frequencies = create_frequency_array(text=text, vocabulary_size=vocabulary_size)
        self.neg_sampling_dist = get_neg_sampling_dist(self.frequencies)

        self.loss_sum = 0.0
        self.loss_count = 0
        self.window_size = 10_000
        self.ema_loss = None
        self.ema_alpha = 0.001

    def train_epoch(self, batch_size: int, learning_rate: float) -> None:

        batch = []

        for center_word, context_word in generate_pairs(self.text, self.window_radius):
            batch.append((center_word, context_word))

            if len(batch) == batch_size:
                self._train_batch(batch, learning_rate)
                batch.clear()

        # handle remainder
        if batch:
            self._train_batch(batch, learning_rate)

        self.save_checkpoint()
        print(f"Epoch done. EMA loss = {self.ema_loss:.4f}")

    def _train_batch(self, batch, learning_rate):
        for center_word, context_word in batch:
            neg_samples = sample_negative_contexts(
                self.negative_sample_size,
                self.neg_sampling_dist,
                forbidden=context_word
            )

            v_c, u_o, u_ni = self.embedder.forward(
                center_word,
                context_word,
                neg_samples
            )

            loss = self.embedder.backward(
                learning_rate,
                v_c,
                u_o,
                u_ni,
                center_word,
                context_word,
                neg_samples
            )

            self._update_trackers(loss)

    def _update_trackers(self, loss):
        self.loss_sum += loss
        self.loss_count += 1

        # EMA
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = (
                self.ema_alpha * loss
                + (1 - self.ema_alpha) * self.ema_loss
            )

        # Sliding window log
        if self.loss_count % self.window_size == 0:
            mean_loss = self.loss_sum / self.window_size
            print(f"Step {self.loss_count}: mean loss = {mean_loss:.4f}, EMA = {self.ema_loss:.4f}")
            self.loss_sum = 0.0

    def save_checkpoint(self) -> None:
        np.save(self.run.run_path / "input_weights.npy", self.embedder.input_weights)
        np.save(self.run.run_path / "output_weights.npy", self.embedder.output_weights)
