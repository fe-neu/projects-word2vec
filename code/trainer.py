import numpy as np
import time
import logging
from logging import Logger

from embedder import Embedder, sigmoid
from data_utils import generate_pairs
from negative_sampling import *
from run import Run

class Trainer:
    def __init__(self,
                 embedder: Embedder,
                 run: Run,
                 text: list[list[int | None]],
                 window_radius: int,
                 vocabulary_size: int,
                 negative_sample_size: int,
                 eval_steps: int,
                 save_steps: int,
                 logger: Logger = None
                 ):
        
        self.embedder = embedder
        self.run = run
        self.text = text
        self.window_radius = window_radius
        self.negative_sample_size = negative_sample_size
        self.logger = logger

        self.frequencies = create_frequency_array(text=text, vocabulary_size=vocabulary_size)
        self.neg_sampling_dist = get_neg_sampling_dist(self.frequencies)

        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.loss_sum = 0.0
        self.step_count = 0
        self.ema_loss = None
        self.ema_alpha = 0.001
        self.last_eval_time = time.perf_counter()
        self.last_eval_step = 0

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

            self.step_count += 1
            self._update_trackers(loss)

            if self.step_count % self.save_steps == 0: self.save_checkpoint()

    def _update_trackers(self, loss):
        self.loss_sum += loss

        # EMA
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = (
                self.ema_alpha * loss
                + (1 - self.ema_alpha) * self.ema_loss
            )

        # Sliding window log
        if self.step_count % self.eval_steps == 0:
            now = time.perf_counter()

            steps_since_last = self.step_count - self.last_eval_step
            elapsed = now - self.last_eval_time
            steps_per_sec = steps_since_last / elapsed if elapsed > 0 else float("inf")

            mean_loss = self.loss_sum / self.eval_steps

            out_str = f"Step {self.step_count} | mean loss = {mean_loss:.4f} | EMA = {self.ema_loss:.4f} | {steps_per_sec:.1f} steps/s"
            
            if self.logger: self.logger.info(out_str)
            else: print(out_str)

            self.loss_sum = 0.0
            self.last_eval_time = now
            self.last_eval_step = self.step_count


    def save_checkpoint(self) -> None:
        self.logger.info("Creating Checkpoint...")
        np.save(self.run.run_path / "input_weights.npy", self.embedder.input_weights)
        np.save(self.run.run_path / "output_weights.npy", self.embedder.output_weights)
