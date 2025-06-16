# EW-GCN
Entity-Word Graph Convolutional Network (EW-GCN)

I implement an Entity-Word Graph Convolutional Network (EW-GCN) for text classification using the 20-Newsgroups dataset. It builds graphs from documents, where nodes are entities and content words, and uses graph convolutional layers to classify texts based on entity-focused features.
## Requirements

* Python 3.x
* Required libraries are listed in `requirements.txt`.

## Steps
1. Install dependencies with:
```python
pip install -r requirements.txt
```

2. Download the spaCy model:
```
python -m spacy download en_core_web_md
```

## Dataset
The 20-Newsgroups dataset is fetched automatically via scikit-learn. It includes 18,846 documents across 20 categories, with headers, footers, and quotes removed.

## How to Run

1. Setup:

* Install dependencies using `requirements.txt`.
* Download the spaCy model as shown above.


2. Execute:

Run the script:

```
python ewgcn.py
```

* The script will:
  - Download the dataset.
  - Build document graphs.
  - Train the model for 50 epochs.
  - Display training loss and validation accuracy per epoch.





## Overview

**Graph Construction**: Documents are converted into graphs with entities and words as nodes, connected by entity-word and word-word edges.

**Model**: The EW-GCN uses two GCN layers, entity-focused pooling, and a linear classifier.

**Training**: Uses Adam optimizer (learning rate 0.001, weight decay 0.0005).

## Notes

- Documents without entities are skipped.
- Node features use half-precision (float16) embeddings for efficiency.
- Validation accuracy is computed after each epoch.

