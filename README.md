# HemaGraph: Breaking Barriers in Hematologic Single Cell Classification with Graph Attention

This repository contains the PyTorch/PyG implementation of Hemagraph, a graph attention network model for multi-class classification of hematologic cell populations from single-cell flow cytometry data, as presented in the paper "Hemagraph: Breaking Barriers in Hematologic Single Cell Classification with Graph Attention Networks".

## Model Architecture

The model architecture follows the graph attention network framework described in the original GAT paper by Velickovic et al. The key components are:

- Graph attention layers that compute attention coefficients between node features using a self-attention mechanism. This allows the model to assign different importance scores to different nodes in a neighborhood.
- Multi-head attention with 8 attention heads to capture different representations subspaces.
- A weighted negative log-likelihood loss function to handle a class imbalance in the highly skewed dataset.

## Data

The data consists of flow cytometry measurements from bone marrow samples of 30 patients, with approximately hundreds of thousands of cells per patient and 12 features per cell. The features include forward/side scattering, fluorescent marker intensities, etc.

The nodes represent individual cells, and the edges connect a node to its 7 nearest neighbors. Graphs are fully connected locally within the neighborhood and can have up to more than one million edges.

There are 5 classes representing different cell types - T cells, B cells, monocytes, Mast cells, and Hematopoietic cells. The classes are highly imbalanced, with some types occurring at frequencies below 0.01%.

## Training Modes

The model is trained in two modes:

1. Inductive learning - standard supervised training and evaluation on separate train/validation/test graphs.
2. Transductive learning - semi-supervised training on a single large graph with 50% unlabeled nodes.

## Installation

The code was developed with:

- Python 3.8
- PyTorch 1.7

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The main model training scripts are:

- `inductive/train_inductive.py` - model training with inductive learning.
- `trans/train_transductive.py` - model training with transductive learning.
- `utils/fcs2knn.py` - kNN graphs from flow cytometry data.
- `utils/mask.py` - masks and weights generation for taking into account the strong class imbalance.

## Supplementary Material
- `suppl/HemaGraph_suppl.pdf`: Supplementary explanations about our patient dataset.

## Citation
If you find this repository useful, please consider citing the paper:

```bash
@article{bini2024hemagraph,
  title={HemaGraph: Breaking Barriers in Hematologic Single Cell Classification with Graph Attention},
  author={Bini, Lorenzo and Mojarrad, Fatemeh Nassajian and Matthes, Thomas and Marchand-Maillet, St{\'e}phane},
  journal={arXiv preprint arXiv:2402.18611},
  year={2024}
}
```

## Contact

For any questions, please contact the authors or open an issue on GitHub. Data can be available upon request.
