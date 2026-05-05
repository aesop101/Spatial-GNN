# Dataset: Region2Vec / Spatial-GNN

This directory contains the spatial and interaction data used for training and evaluating the Spatial-GNN model.

## File Descriptions

- **`feature_matrix_f1.csv`**: Node attribute matrix containing socioeconomic and demographic features for each region.
- **`Flow_matrix.csv`**: Matrix representing the flow of people/goods between pairs of regions.
- **`Spatial_matrix.csv`**: Adjacency matrix defining the connectivity between regions.
- **`Spatial_distance_matrix.csv`**: Pairwise spatial distances between region centroids.
- **`feature_matrix_lwinc.csv`**: Income percentile data used specifically for calculating homogeneity scores and evaluating clustering quality.
- **`flow_reID.csv`**: Processed flow data used for intra/inter-community flow analysis.
- **`spatial_matrix_rook.csv`**: Connectivity matrix based on the Rook contiguity principle.

## Source & Citation

The data in this folder is based on the research presented in:

**Region2Vec: Community Detection on Spatial Networks Using Graph Embedding with Node Attributes and Spatial Interactions**
*Yunlei Liang, Jiawei Zhu, Wen Ye, Song Gao*

If you use this data in your research, please ensure you cite the original authors.
