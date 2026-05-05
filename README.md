# Spatial-GNN

Application of Graph Attention Network (GAT) for spatial use.

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
- `src/model.py`: SpatialGAT implementation.
- `src/train.py`: Training pipeline.
- `src/data_loader.py`: Data loading.
- `src/clustering.py`: Agglomerative clustering on learned embeddings.

## References
- **Region2Vec: Community Detection on Spatial Networks Using Graph Embedding with Node Attributes and Spatial Interactions**
  *Yunlei Liang, Jiawei Zhu, Wen Ye, Song Gao*
