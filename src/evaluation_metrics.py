import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import os
from dotenv import load_dotenv

load_dotenv()
INCOME_PERCENTILE_FILE = os.getenv("INCOME_PERCENTILE_FILE")
EPS = 1e-15
FEATURES_FIN_FILE = os.getenv("FEATURES_FIN_FILE")
FEATURES_FILE = os.getenv("FEATURES_FILE")
FLOW_RATE_FILE = os.getenv("FLOW_RATE_FILE")


def similarity(labels, n_clusters, feature_path=FEATURES_FILE):
    features = np.loadtxt(feature_path, delimiter=',') 
    X = features[:, 1:]

    sim_scores = []
    dist_scores = []

    for c in range(n_clusters):
        indices = np.where(labels == c)[0]
        
        if len(indices) < 2:
            continue  
            
        X_cluster = X[indices]
        cossim_com = cosine_similarity(X_cluster)
        eucdist_com = euclidean_distances(X_cluster)

        tri_idx = np.triu_indices(len(indices), k=1)
        
        sim_scores.append(np.mean(cossim_com[tri_idx]))
        dist_scores.append(np.mean(eucdist_com[tri_idx]))

    median_sim = np.median(sim_scores) if sim_scores else 0
    median_dist = np.median(dist_scores) if dist_scores else 0

    print(f"The median cosine similarity is {median_sim:.3f}")
    print(f"The median euclidean distance is {median_dist:.3f}")

    return median_sim, median_dist


def cal_inequality(values):
    mean = np.mean(values)
    std = np.std(values)
    ineq = std/math.sqrt(mean*(1-mean))
    return ineq

def community_inequality(labels, file_name, path, k = 13):
    features = np.loadtxt(FEATURES_FIN_FILE, delimiter=',') #use updated features   
    features = features[:,1:]
    pdist = np.linalg.norm(features[:, None]-features, ord = 2, axis=2)

    ineq_dict = {}
    for c in range(k):
        ct_com = np.where(labels == c)[0]
        if len(ct_com) < 2:
            continue
        else:
            pdist_com = pdist[ct_com[:,None], ct_com[None,:]]  #slice the pdist so all the included values is for this community
            dist = pdist_com[np.triu_indices(len(ct_com), k = 1)]
            
            ineq = cal_inequality(dist)
            ineq_dict[c] = ineq

    median_ineq = np.median(list(ineq_dict.values()))
    print("The median inequality is {:.3f}".format(median_ineq))

    return median_ineq


def get_homogenity_score(labels, income_file=INCOME_PERCENTILE_FILE):
    X_lwinc = np.loadtxt(income_file, delimiter=',', usecols=range(1, 2)).flatten()
    n_thres = 5
    thresholds = np.linspace(0, 1, n_thres + 1)
    
    bin_edges = np.quantile(X_lwinc, thresholds)
    X_classes = np.digitize(X_lwinc, bin_edges[1:-1]) 
    homo_score = metrics.homogeneity_score(X_classes, labels)
    
    print(f"The homogeneous score is {homo_score:.3f}")
    return homo_score


def intra_inter_idx(labels):
    flow = pd.read_csv(FLOW_RATE_FILE)
    if 'Unnamed: 0' in flow.columns:
        flow = flow.drop(columns='Unnamed: 0')

    flow['From_Com'] = np.take(labels, flow['From'].values)
    flow['To_Com'] = np.take(labels, flow['To'].values)

    com_stats = flow.groupby(['From_Com', 'To_Com'])['visitor_flows'].sum().reset_index()
    is_intra = com_stats['From_Com'] == com_stats['To_Com']
    
    intra_sum = com_stats.loc[is_intra, 'visitor_flows'].sum()
    inter_sum = com_stats.loc[~is_intra, 'visitor_flows'].sum()
    
    total_ratio = intra_sum / inter_sum if inter_sum != 0 else np.nan
    
    print(f"The total intra/inter ratio is {total_ratio:.3f}")
    return total_ratio