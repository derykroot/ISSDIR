# ISSDiR: Inductive Self-Supervised Dimensionality Reduction

This repository contains the implementation of the **ISSDiR (Inductive Self-Supervised Dimensionality Reduction)** method, proposed for dimensionality reduction for tasks in Content-Based Image Retrieval (CBIR) systems.

---

## Overview

ISSDiR is an approach that combines:
- **Dimensionality reduction** with self-supervised learning.
- **Generalization to unseen data**, more discriminative, low-dimensional representations for image retrieval.
- A **hybrid loss function** combining cross-entropy and contrastive loss, with a weighting factor based on intercluster distances.

The main goal is to create compact and discriminative representations, while maintaining the ability to generalize to expanding datasets.

---

## Features

- Feature extraction using pre-trained networks.
- Clustering with pseudo-labels.
- Dimensionality reduction using a neural network.
- Ranked list generation for retrieval tasks.

---

## Requirements

The project requires the following dependencies:

- **Python 3.11+**
- **Core Libraries**:
  - PyTorch 2.1
  - umap-pytorch

Install the dependencies using:

```bash
pip install torch umap-pytorch
```

---

## Usage

All experiments are implemented in Jupyter Notebooks.

## Results

Experiments were conducted on popular datasets such as:

- MNIST
- CIFAR-10
- FashionMNIST
- Corel5K

Results show that ISSDiR delivers competitive performance, with significant improvements in retrieval tasks when compared to traditional dimensionality reduction methods.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository.
2. Create a feature branch: `git checkout -b my-feature`.
3. Commit your changes: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin my-feature`.
5. Open a Pull Request.


---

## Authors

- Deryk Willyan Biotto ([GitHub](https://github.com/derykbiotto))
- Guilherme Henrique Jardim
- Vinicius Atsushi Sato Kawai
- Bionda Rozin
- Denis Henrique Pinheiro Salvadeo
- Daniel Carlos Guimarães Pedronette


