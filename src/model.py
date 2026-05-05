import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpatialGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        N = h.size(0)
        Wh = self.W(h)

        e = Wh.repeat_interleave(N, dim=0).view(N, N, self.out_features)
        e = e + Wh.repeat(N, 1).view(N, N, self.out_features)
        e = self.leakyrelu(e)
        
        attention = torch.matmul(e, self.a).squeeze(2)
        spatial_prior = torch.log(adj + 1e-10) 
        attention = attention + spatial_prior

        mask = (adj <= 0)
        attention = attention.masked_fill(mask, -9e15)
        
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.mm(attention, Wh)

        return F.elu(h_prime) if self.concat else h_prime

class SpatialGAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, alpha=0.2, nheads=4):
        super(SpatialGAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([
            SpatialGATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
            for _ in range(nheads)
        ])
        
        self.out_att = SpatialGATLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)
        self.res_proj = nn.Linear(nfeat, nout)
        self.norm = nn.LayerNorm(nout)

    def forward(self, x, adj):
        x_in = x
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        
        x = self.norm(x + self.res_proj(x_in))
        return x, F.softmax(x, dim=1)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()
        self.model = SpatialGAT(nfeat, nhid, nout, dropout)

    def forward(self, x, adj):
        return self.model(x, adj)