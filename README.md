# Spatial-GNN

A high-performance Graph Attention Network (GAT) with spatial priors for region embedding.

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Update `.env` with your local paths. Ensure `DATA_DIR` points to the `data` folder.

3. **Run Training**:
   ```bash
   cd src
   python train.py
   ```

## Key Components
- `src/model.py`: SpatialGAT implementation (Multi-head attention + Spatial Priors).
- `src/train.py`: Training pipeline with InfoNCE loss and Laplacian regularization.
- `src/data_loader.py`: Data loading with Laplacian Positional Encodings (LPE).
- `src/clustering.py`: Agglomerative clustering on learned embeddings.

## Data Requirements
Place the following in the `data/` directory:
- `feature_matrix_f1.csv`, `Flow_matrix.csv`, `Spatial_matrix.csv`, `Spatial_distance_matrix.csv`

## References
If you use this code or data, please cite the following paper:
- **Region2Vec: Community Detection on Spatial Networks Using Graph Embedding with Node Attributes and Spatial Interactions**
  *Yunlei Liang, Jiawei Zhu, Wen Ye, Song Gao*
