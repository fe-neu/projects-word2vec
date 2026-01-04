import json
from pathlib import Path

from run import Run
from trainer import Trainer
from embedder import Embedder
from data_preprocessing import *
from logging import getLogger
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

run_name = "Third_Run"
run_description = "Testing"

frequency_threshold = 315
embed_dims = 128
window_radius = 3
negative_sample_size = 10

batch_size = 32
epochs = 3
learning_rate = 0.001

def main():
    logger = getLogger()

    logger.info("Creating new Run...")
    run = Run(run_name=run_name, description=run_description)

    logger.info("Loading Data...")
    data = load_data()[:100000]

    logger.info(f"Creating Vocabulary with threshold {frequency_threshold}...")
    vocabulary = create_vocabulary(text=data, threshold=frequency_threshold)
    logger.info(f"Vocaublary Size is {len(vocabulary)}")

    logger.info(f"Creating Vocabulary Map...")
    vocabulary_map = create_vocabulary_map(vocabulary)

    logger.info(f"Saving vocabulary...")
    with run.run_path.joinpath("vocabulary_map.json").open("w", encoding="utf-8") as f:
        json.dump(vocabulary_map, f, indent=2, ensure_ascii=False)

    logger.info(f"Encoding Text...")
    encoded_text = encode_text(text=data, vocabulary_map=vocabulary_map)

    logger.info(f"Creating Embedder...")
    embedder = Embedder(
        embed_dims=embed_dims,
        vocabulary_size=len(vocabulary),
        random_seed=42
    )

    logger.info(f"Creating Trainer...")
    trainer = Trainer(
        embedder=embedder,
        run=run,
        text=encoded_text,
        window_radius=window_radius,
        vocabulary_size=len(vocabulary),
        negative_sample_size=negative_sample_size
    )

    for epoch_n in range(epochs):
        logger.info(f"Now starting Epoch {epoch_n+1}")
        trainer.train_epoch(batch_size=batch_size, learning_rate=learning_rate)

    logger.info(f"Done")

if __name__ == "__main__":
    main()


