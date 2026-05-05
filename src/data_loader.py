import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import os
import re
import scipy.sparse as sp
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
EPS = 1e-15


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot



def get_laplacian_eigenvectors(adj, k=8):
    if torch.is_tensor(adj):
        adj = adj.cpu().numpy()
    
    deg = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(deg, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    adj_sym = (adj + adj.T) / 2
    I = np.eye(adj.shape[0])
    laplacian = I - (d_inv_sqrt[:, np.newaxis] * adj_sym * d_inv_sqrt[np.newaxis, :])
    
    try:
        vals, vecs = np.linalg.eigh(laplacian)
        return torch.FloatTensor(vecs[:, 1:k+1])
    except Exception as e:
        print(f"Warning: Laplacian Eigendecomposition failed: {e}, returning zeros.")
        return torch.zeros((adj.shape[0], k))

def load_widata(path=DATA_DIR, dataset="wi", hops=5, k_eig=8):
    print('Loading {} dataset...'.format(dataset))

    print(" - Loading features...")
    features = pd.read_csv(path+'\\feature_matrix_f1.csv', header=None).values
    features = torch.FloatTensor(features[:,1:])

    print(" - Loading spatial adjacency...")
    spatial_adj = pd.read_csv(path + '\\Spatial_matrix.csv', header=None).values
    
    print(" - Computing Laplacian Eigenvectors...")
    pos_enc = get_laplacian_eigenvectors(spatial_adj, k=k_eig)
    features = torch.cat([features, pos_enc], dim=1)

    print(" - Loading flow matrix...")
    intensity_m = pd.read_csv(path + '\\Flow_matrix.csv', header=None).values
    intensity_neg = np.zeros([len(intensity_m),len(intensity_m)])
    intensity_neg[intensity_m == 0] = 1
    intensity_pos = np.zeros([len(intensity_m),len(intensity_m)])
    intensity_pos[intensity_m > 0] = 1

    intensity_m = np.log(intensity_m + EPS)
    intensity_m_norm = mx_normalize(intensity_m)
    intensity_m_norm = torch.FloatTensor(intensity_m_norm)
    strength = torch.sum(intensity_m_norm, axis = 0)
    strength = torch.FloatTensor(strength)

    print(" - Loading spatial distance matrix...")
    hops_m = pd.read_csv(path + '\\Spatial_distance_matrix.csv', header=None).values
    zero_entries = hops_m < hops
    hops_m = 1/(np.log(hops_m + EPS)+1)
    hops_m[zero_entries] = 0
    hops_m = torch.FloatTensor(hops_m)

    intensity_m = torch.FloatTensor(intensity_m)
    intensity_neg = torch.FloatTensor(intensity_neg)
    intensity_pos = torch.FloatTensor(intensity_pos)

    adj = normalize(spatial_adj + sp.eye(spatial_adj.shape[0]))
    adj = torch.FloatTensor(adj)

    print(" - Data loading complete.")
    return adj, features, intensity_m, intensity_neg, intensity_pos, hops_m, intensity_m_norm, strength

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def mx_normalize(mx):
    mx_std = (mx - mx.min()) / (mx.max() - mx.min())
    return mx_std

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def purge(dir, filename, best_epoch, spill_num):
    del_list = ['Epoch_{}_'.format(i) + filename for i in range(0, best_epoch)]
    if spill_num > 0:
        tmp = ['Epoch_{}_'.format(j) + filename for j in range(best_epoch + 1, best_epoch + spill_num + 1)]
        del_list.extend(tmp)        
    for f in os.listdir(dir):
        if f in del_list:
            os.remove(os.path.join(dir,f))