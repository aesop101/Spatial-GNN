from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from data_loader import load_widata, purge
from model import GCN
from clustering import run_aggclustering
import csv
import os
import math
from dotenv import load_dotenv
load_dotenv()

NO_CUDA=True if torch.cuda.is_available() else False
SEED=int(os.getenv("SEED", 42))
EPOCHS=int(os.getenv("EPOCHS", 10))
PATIENCE=int(os.getenv("PATIENCE", 50))
LTYPE=os.getenv("LTYPE", 'divreg')
LR=float(os.getenv("LR", 0.001))
WEIGHT_DECAY=float(os.getenv("WEIGHT_DECAY", 5e-4))
HIDDEN=int(os.getenv("HIDDEN", 16))
OUTPUT=int(os.getenv("OUTPUT", 14))
DROPOUT=float(os.getenv("DROPOUT", 0.1))
HOPS=int(os.getenv("HOPS", 5))
RESULT_PATH=os.getenv("RESULT_PATH", '../result')

np.random.seed(SEED)
torch.manual_seed(SEED)
if NO_CUDA:
    torch.cuda.manual_seed(SEED)


adj, features, labels, neg_mask, pos_mask, hops_m, intensity_m_norm, strength = load_widata()
model = GCN(nfeat=features.shape[1],
            nhid=HIDDEN,
            nout=OUTPUT,
            dropout=DROPOUT)
optimizer = optim.Adam(model.parameters(),
                       lr=LR, weight_decay=WEIGHT_DECAY)
EPS = 1e-15

if NO_CUDA:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    neg_mask = neg_mask.cuda()
    pos_mask = pos_mask.cuda()
    hops_m = hops_m.cuda()

def info_nce_loss(query, pos, neg, temperature=0.1):
    query = F.normalize(query, dim=1)
    pos = F.normalize(pos, dim=2)
    neg = F.normalize(neg, dim=2)
    
    pos_sim = torch.sum(query.unsqueeze(1) * pos, dim=2)
    neg_sim = torch.sum(query.unsqueeze(1) * neg, dim=2)
    exp_pos = torch.exp(pos_sim / temperature)
    exp_neg_sum = torch.exp(neg_sim / temperature).sum(dim=1, keepdim=True)
    
    loss = -torch.log(exp_pos / (exp_pos + exp_neg_sum + EPS)).mean()
    return loss


def improved_spatial_loss(query, pos, neg, adj, temperature=0.1, alpha=0.5):
    base_loss = info_nce_loss(query, pos, neg, temperature)
    query_norm = F.normalize(query, dim=1)
    d = torch.diag(torch.sum(adj, dim=1))
    laplacian = d - adj

    spatial_reg = torch.trace(torch.mm(torch.mm(query_norm.t(), laplacian), query_norm)) / query.shape[0]
    return base_loss + (alpha * spatial_reg)

def sample_batch(intensity_pos, intensity_neg, hops_m, num_pos=5, num_neg=15, num_hard=5):
    N = intensity_pos.shape[0]
    pos_indices = []
    neg_indices = []
    
    for i in range(N):
        pos_candidates = torch.where(intensity_pos[i] > 0)[0]
        if len(pos_candidates) > 0:
            pos_idx = pos_candidates[torch.randint(0, len(pos_candidates), (num_pos,))]
        else:
            pos_idx = torch.full((num_pos,), i, device=intensity_pos.device)
        pos_indices.append(pos_idx)
        neg_candidates = torch.where(intensity_neg[i] > 0)[0]
        if len(neg_candidates) > 0:
            rand_neg = neg_candidates[torch.randint(0, len(neg_candidates), (num_neg,))]
            if num_hard > 0:
                i_hops = hops_m[i] * intensity_neg[i] 
                _, hard_idx_top = torch.topk(i_hops, k=min(num_hard * 2, len(neg_candidates)))
                hard_neg = hard_idx_top[torch.randint(0, len(hard_idx_top), (num_hard,))]
                neg_idx = torch.cat([rand_neg, hard_neg])
            else:
                neg_idx = rand_neg
        else:
            neg_idx = torch.randint(0, N, (num_neg + num_hard,), device=intensity_pos.device)
            
        neg_indices.append(neg_idx)
        
    return torch.stack(pos_indices), torch.stack(neg_indices)

t_total = time.time()
loss_list = []

NUM_POS = 5
NUM_NEG = 15
NUM_HARD = 5
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))

for epoch in range(EPOCHS):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    
    output, _ = model(features, adj)
    pos_idx, neg_idx = sample_batch(pos_mask, neg_mask, hops_m, num_pos=NUM_POS, num_neg=NUM_NEG, num_hard=NUM_HARD)
    pos_emb = output[pos_idx] 
    neg_emb = output[neg_idx] 
    loss_train = improved_spatial_loss(output, pos_emb, neg_emb, adj, alpha=0.1)
    
    loss_train.backward()
    optimizer.step()
    
    loss_list.append(loss_train.item())

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.5f}'.format(loss_train.item()),
          'time: {:.4f}s'.format(time.time() - t))

    result_path = RESULT_PATH
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if epoch >= 50:
        save_name = 'lr_{}_dropout_{}_hidden_{}_output_{}_patience_{}_hos_{}_losstype_{}_seed_{}.csv'.format(LR, DROPOUT, HIDDEN, OUTPUT, PATIENCE, HOPS, LTYPE, SEED)
        np.savetxt(os.path.join(result_path, 'Epoch_{}_'.format(epoch) + save_name), output.detach().cpu().numpy())

        if epoch > 50 + PATIENCE and loss_train > np.average(loss_list[-PATIENCE:]):
            best_epoch = loss_list.index(min(loss_list))
            print('Lose patience, stop training...')
            print('Best epoch: {}'.format(best_epoch))
            purge(result_path, save_name, best_epoch, epoch-best_epoch)
            break

        if epoch == EPOCHS -1:
            print('Last epoch, saving...')
            best_epoch = epoch
            purge(result_path, save_name, best_epoch, 0)


print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


result_csv = os.path.join(RESULT_PATH, 'result.csv')
if not os.path.exists(result_csv):
    with open(result_csv, 'w') as f:
        csv_write = csv.writer(f)
        csv_head = ['epoch', 'losstype', 'hops', 'inertia', 'total_ratio', 'global_ineq', 'output', 'hidden', 'lr', 'dropout', 'patience', 'median_ineq','n_clu']
        csv_write.writerow(csv_head)

print(f"epochs = {EPOCHS}")
print(f"best epoch: {best_epoch}")
if 'best_epoch' in locals() and best_epoch >= 50:
    print("Running evaluation on best epoch: {}".format(best_epoch))
    best_file = 'Epoch_{}_'.format(best_epoch) + save_name
    labels, total_ratio, median_ineq, median_sim, median_dist, homo_score = run_aggclustering(result_path, best_file, 'euclidean', OUTPUT)
    res_row = [best_epoch, LTYPE, HOPS, None, total_ratio, None, OUTPUT, HIDDEN, LR, DROPOUT, PATIENCE, median_ineq, OUTPUT]

    if not os.path.exists(result_path):
        os.mkdir(result_path)
        
    with open(os.path.join(result_path, 'result.csv'), 'a', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(res_row)
    print("Results appended to {}".format(os.path.join(result_path, 'result.csv')))
else:
    print("No evaluation performed (best_epoch < 5 or not defined).")
