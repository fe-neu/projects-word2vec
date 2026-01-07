# Word2Vec (Skip-Gram with Negative Sampling) – NumPy Implementation

This project is a minimal, educational implementation of the **Word2Vec skip-gram model with negative sampling**, written entirely in **NumPy**.

It is inspired by the original Word2Vec paper:

> Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013).
> *Distributed Representations of Words and Phrases and their Compositionality*.
> Advances in Neural Information Processing Systems (NeurIPS).

The goal of this project is **clarity and transparency**, not performance or feature completeness.

---

## Project Overview

The implementation covers the full Word2Vec training pipeline:

* Text preprocessing and vocabulary construction
* Skip-gram pair generation
* Negative sampling
* Training loop with EMA loss tracking
* Saving and loading trained embeddings
* Nearest-neighbor queries using cosine similarity or Euclidean distance

The entire model is implemented **without PyTorch or TensorFlow** to make the underlying mechanics explicit.

---

## Dependencies

* Python 3.10+
* NumPy
* KaggleHub (for dataset download)

No deep learning frameworks are used.

---

## Dataset

The training data is the **Simple English Wikipedia** corpus, downloaded automatically from Kaggle:

[https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish](https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish)

The dataset is downloaded on first use and cached locally.

---

## Project Structure

```
code/
├── data_utils.py              # Text loading, cleaning, encoding, skip-gram pairs
├── embedder.py                # Skip-gram embedding model + gradients
├── embedding_retriever.py     # Nearest-neighbor queries on trained embeddings
├── negative_sampling.py       # Frequency counts & negative sampling
├── trainer.py                 # Training loop and logging
├── run.py                     # Run directory & checkpoint management
├── main.py                    # Entry point (configure and start training)
├── runs/                      # Saved runs and embeddings
├── environment.yml            # Conda environment definition
└── README.md
```

---

## How to Run

### 1. Create the Conda Environment

From the project root:

```
conda env create -f environment.yml
conda activate word2vec
```

(Adjust the environment name if you changed it.)

---

### 2. Navigate to the Code Directory

```
cd code
```

---

### 3. Configure Training Parameters

Open `main.py` and adjust parameters such as:

* embedding dimension
* context window size
* negative sample size
* learning rate
* batch size
* number of epochs

---

### 4. Start Training

Run:

```
python main.py
```

During training, you will see periodic logs including:

* Mean loss
* Exponential moving average (EMA) loss
* Training speed (steps per second)

Trained embeddings are saved automatically in the `runs/` directory.

---

## Using Trained Embeddings

After training, embeddings can be loaded using `EmbeddingRetriever` to:

* Retrieve word vectors
* Find nearest neighbors
* Compare cosine similarity vs. Euclidean distance

This is useful for inspecting semantic structure and debugging training behavior.

---

## Notes

* This implementation prioritizes **readability over speed**
* No subsampling, hierarchical softmax, or multi-threading
* Intended for learning, experimentation, and inspection

---

## Notebook

A small Jupyter notebook (`explore_embeddings.ipynb`) is included to interactively
inspect trained embeddings. It demonstrates nearest-neighbor queries, similarity
metrics, and simple vector arithmetic using the existing Python APIs.

The notebook contains **no training logic** and is intended purely for exploration.

---

## License

This project is intended for educational use.
