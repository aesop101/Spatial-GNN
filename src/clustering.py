import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from dotenv import load_dotenv

from evaluation_metrics import (
    similarity, community_inequality, 
    get_homogenity_score, intra_inter_idx
)

load_dotenv()
SPATIAL_MATRIX_PATH = Path(os.getenv('SPATIAL_MATRIX_ROOK_FILE', ''))

def run_aggclustering(data_path, file_name, affinity, n_clusters, linkage='ward'):
    data_path = Path(data_path)
    file_path = data_path / file_name
    
    X = pd.read_csv(file_path, delimiter=' ', header=None).values
    if SPATIAL_MATRIX_PATH.exists():
        adj = pd.read_csv(SPATIAL_MATRIX_PATH, delimiter=',', header=None).values
        adj = sparse.csr_matrix(adj)
    else:
        adj = None
        print("Warning: Spatial matrix file not found. Running without connectivity.")

    model = AgglomerativeClustering(
        n_clusters=n_clusters, 
        connectivity=adj, 
        metric=affinity, 
        linkage=linkage
    )

    labels = model.fit_predict(X)
    clean_name = file_name.replace('.csv', '')
    results = {
        "labels": labels,
        "homo_score": get_homogenity_score(labels),
        "total_ratio": intra_inter_idx(labels),
        "median_ineq": community_inequality(labels, clean_name, str(data_path), n_clusters),
    }
    
    median_sim, median_dist = similarity(labels, n_clusters)
    
    return (
        results["labels"], 
        results["total_ratio"], 
        results["median_ineq"], 
        median_sim, 
        median_dist, 
        results["homo_score"]
    )