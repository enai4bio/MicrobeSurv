import dgl
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from SAGPool import SAGPool
import dgl.nn.pytorch as dglnn
from torch.functional import F
from MVPool import GCN, MVPool
from pycox.models.loss import CoxPHLoss
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class PartialHazard(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, out_dim, bias=False),
            nn.Sigmoid() if out_dim != 1 else nn.Identity()
        )
    
    def forward(self, x):
        return self.mlp(x)

class MLP2(nn.Module):
    def __init__(self, in_dim, embed_dim, haz_dim, out_dim, dropout):
        super().__init__()

        self.embed_layer = nn.Sequential(
            nn.Linear(in_dim, embed_dim, bias=False),
            nn.ReLU()
        )

        self.partial_hazard = PartialHazard(embed_dim, haz_dim, out_dim, dropout)

    def forward(self, x):
        embed = self.embed_layer(x)
        return self.partial_hazard(embed)

class MLP(nn.Module):
    def __init__(self, c_in_dim, g_in_dim, c_embed_dim, g_embed_dim, g_hid_layer, haz_dim, out_dim, dropout):
        super().__init__()
        self.c_embed_layer = None
        self.g_embed_layer = None
        self.c_in_dim = c_in_dim

        fusion_dim = 0
        if c_in_dim:
            self.c_embed_layer = nn.Sequential(
                nn.Linear(c_in_dim, c_embed_dim, bias=False),
                nn.ReLU()
            )
            fusion_dim += c_embed_dim

        if g_in_dim:
            layers = [g_in_dim] + [g_embed_dim] * g_hid_layer
            lins = []
            for i in range(g_hid_layer):
                lins.append(nn.Linear(layers[i], layers[i+1], bias=False))
                lins.append(nn.ReLU())
            self.g_embed_layer = nn.Sequential(*lins)
            fusion_dim += g_embed_dim

        self.partial_hazard = PartialHazard(fusion_dim, haz_dim, out_dim, dropout)

    def forward(self, x):
        arr = []
        if self.c_embed_layer:
            c_embed = self.c_embed_layer(x[:,:self.c_in_dim])
            arr.append(c_embed)

        if self.g_embed_layer:
            g_embed = self.g_embed_layer(x[:,self.c_in_dim:])
            arr.append(g_embed)

        embed = torch.cat(arr, 1)
        return self.partial_hazard(embed)

class CoxModel:
    def __init__(self, net, lr):
        self.net = net
        self.base_haz = None
        self.base_cum_haz = None # Tx1
        self.loss = CoxPHLoss()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
    
    def compute_haz(self, inps, y):
        if type(inps) is not tuple: inps = (inps,)
        base_haz = y.assign(expg=np.exp(self.net(*inps).cpu())) \
                    .groupby('time') \
                    .agg({'expg': 'sum', 'event': 'sum'}) \
                    .sort_index(ascending=False) \
                    .assign(expg=lambda x: x['expg'].cumsum()) \
                    .pipe(lambda x: x['event']/x['expg']) \
                    .fillna(0.).iloc[::-1]
        self.base_haz = base_haz
        self.base_cum_haz = base_haz.cumsum()
    
    def pred_surv(self, inps):
        if type(inps) is not tuple: inps = (inps,)
        expg = np.exp(self.net(*inps).cpu().reshape(1, -1)) # 1xN
        cumulative_hazards = pd.DataFrame(self.base_cum_haz.values.reshape(-1, 1).dot(expg), index=self.base_cum_haz.index)
        surv = np.exp(-cumulative_hazards)
        return surv

    def update(self, inps, yt, ye, trainable=True):
        if type(inps) is not tuple: inps = (inps,)
        p = self.net(*inps)
        loss = self.loss(p, yt, ye)
        if torch.isnan(loss): return None
        if trainable:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return loss.item()

class NonCoxModel:
    def __init__(self, net, lr, times):
        self.net = net
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.times = times
    
    def pred_surv(self, inps):
        if type(inps) is not tuple: inps = (inps,)
        p = self.net(*inps).cpu()
        prob_surv = np.cumprod(p.numpy(), axis=1)
        surv = pd.DataFrame(prob_surv.T, index=self.times)
        return surv

    def update(self, inps, yt, ye, trainable=True):
        if type(inps) is not tuple: inps = (inps,)
        p = self.net(*inps)
        loss = self.loss(p, yt, ye)
        if torch.isnan(loss): return None
        if trainable:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return loss.item()

    def loss(self, pred, yt, ye):
        before_event = yt * pred
        after_event = 1 - ye.unsqueeze(1) * (1 - yt) * pred
        # loss = - torch.log(torch.clamp(torch.cat((before_event, after_event), dim=1), 1e-7))
        loss = - torch.log(torch.clamp(before_event+after_event, 1e-7))
        # breakpoint()
        return loss.mean()

class EarlyStop:
    def __init__(self, step=10) -> None:
        self.cnt = 0
        self.step = step
        self.losses = [np.inf]

    def add(self, loss) -> bool:
        if loss > self.losses[-1]:
            self.cnt += 1
            if self.cnt >= self.step: return True
        else:
            self.cnt = max(0, self.cnt-1)
        self.losses.append(loss)
        return False

class GNN_ori(nn.Module):
    def __init__(self, method, g_in_dim, g_hidden_dim, c_in_dim, c_embed_dim, haz_dim, out_dim, dropout,
                num_heads=2, pool_ratio=0.5,
                readout='both', # max, avg
                fusion='cat' # dot
                ):
        super().__init__()

        self.pool = None
        self.avg_readout = dglnn.AvgPooling()
        self.max_readout = dglnn.MaxPooling()
        self.readout = readout
        self.fusion = fusion

        if method == 'GCN':
            self.conv1 = dglnn.GraphConv(g_in_dim, g_hidden_dim)
            self.conv2 = dglnn.GraphConv(g_hidden_dim, g_hidden_dim)
            self.conv3 = dglnn.GraphConv(g_hidden_dim, g_hidden_dim)
        elif method == 'GAT':
            self.conv1 = dglnn.GATConv(g_in_dim, g_hidden_dim, num_heads=num_heads)
            self.conv2 = dglnn.GATConv(g_hidden_dim, g_hidden_dim, num_heads=num_heads)
            self.conv3 = dglnn.GATConv(g_hidden_dim, g_hidden_dim, num_heads=num_heads)
        elif method == 'GIN':
            lin1 = nn.Linear(g_in_dim, g_hidden_dim)
            lin2 = nn.Linear(g_hidden_dim, g_hidden_dim)
            lin3 = nn.Linear(g_hidden_dim, g_hidden_dim)
            self.conv1 = dglnn.GINConv(lin1, 'sum', learn_eps=True)
            self.conv2 = dglnn.GINConv(lin2, 'sum', learn_eps=True)
            self.conv3 = dglnn.GINConv(lin3, 'sum', learn_eps=True)
        elif method == 'SAG':
            self.conv1 = dglnn.GraphConv(g_in_dim, g_hidden_dim)
            self.conv2 = dglnn.GraphConv(g_hidden_dim, g_hidden_dim)
            self.conv3 = dglnn.GraphConv(g_hidden_dim, g_hidden_dim)
            self.pool = SAGPool(g_hidden_dim, ratio=pool_ratio)

        g_embed_dim = g_hidden_dim * (2 if readout == 'both' else 1)

        if c_in_dim:
            self.embed_layer = nn.Sequential(
                nn.Linear(c_in_dim, c_embed_dim, bias=False),
                nn.ReLU()
            )
            if fusion == 'cat':
                fusion_dim = g_embed_dim + c_embed_dim
            else:
                assert g_embed_dim == c_embed_dim
                fusion_dim = g_embed_dim
        else:
            self.embed_layer = None
            fusion_dim = g_embed_dim

        self.partial_hazard = PartialHazard(fusion_dim, haz_dim, out_dim, dropout)

    def forward(self, inp_c, inp_n, g, return_hg=False):
        embed = self.embed_layer(inp_c) if self.embed_layer else None

        x = F.relu(self.conv1(g, inp_n))
        if len(x.shape) == 3: x = torch.mean(x, dim=1) # GAT
        if self.readout == 'max':
            x1 = self.max_readout(g, x)
        elif self.readout == 'avg':
            x1 = self.avg_readout(g, x)
        elif self.readout == 'both':
            x1 = torch.cat([self.avg_readout(g, x), self.max_readout(g, x)], -1)

        x = F.relu(self.conv2(g, x))
        if len(x.shape) == 3: x = torch.mean(x, dim=1) # GAT
        if self.readout == 'max':
            x2 = self.max_readout(g, x)
        elif self.readout == 'avg':
            x2 = self.avg_readout(g, x)
        elif self.readout == 'both':
            x2 = torch.cat([self.avg_readout(g, x), self.max_readout(g, x)], -1)

        x = F.relu(self.conv3(g, x))
        if len(x.shape) == 3: x = torch.mean(x, dim=1) # GAT
        if self.pool: g, x, _ = self.pool(g, x)
        if self.readout == 'max':
            x3 = self.max_readout(g, x)
        elif self.readout == 'avg':
            x3 = self.avg_readout(g, x)
        elif self.readout == 'both':
            x3 = torch.cat([self.avg_readout(g, x), self.max_readout(g, x)], -1)

        hg = F.relu(x1) + F.relu(x2) + F.relu(x3)
        if return_hg: return hg

        if embed is None:
            fusion = hg
        else:
            if self.fusion == 'cat':
                fusion = torch.cat((embed, hg), 1)
            elif self.fusion == 'dot':
                fusion = embed * hg
        out = self.partial_hazard(fusion)
        return out

class GNN(nn.Module):
    def __init__(self, method, inp_dim, graph, g_dims, c_embed_dim, haz_dim, out_dim, dropout,
                readout='both', # max, avg
                fusion='cat', # dot
                num_heads=2, pool_ratio=0.5,
                ):
        super().__init__()

        self.pool = None
        self.avg_readout = dglnn.AvgPooling()
        self.max_readout = dglnn.MaxPooling()
        self.readout = readout
        self.fusion = fusion
        self.g = graph

        c_in_dim = inp_dim - self.g.num_nodes()
        self.c_in_dim = c_in_dim

        self.convs = []
        if method == 'GCN':
            for i in range(len(g_dims)-1):
                self.convs.append(dglnn.GraphConv(g_dims[i], g_dims[i+1]))
        elif method == 'GAT':
            for i in range(len(g_dims)-1):
                self.convs.append(dglnn.GATConv(g_dims[i], g_dims[i+1], num_heads=num_heads))
        elif method == 'GIN':
            for i in range(len(g_dims)-1):
                lin = nn.Linear(g_dims[i], g_dims[i+1])
                self.convs.append(dglnn.GINConv(lin, 'sum', learn_eps=True))
        elif method == 'SAG':
            for i in range(len(g_dims)-1):
                self.convs.append(dglnn.GraphConv(g_dims[i], g_dims[i+1]))
            self.pool = SAGPool(g_dims[-1], ratio=pool_ratio)
        
        self.convs = nn.ModuleList(self.convs)
        g_embed_dim = g_dims[-1] * (2 if readout == 'both' else 1)

        if c_in_dim:
            self.embed_layer = nn.Sequential(
                nn.Linear(c_in_dim, c_embed_dim, bias=False),
                nn.ReLU()
            )
            if fusion == 'cat':
                fusion_dim = g_embed_dim + c_embed_dim
            else:
                assert g_embed_dim == c_embed_dim
                fusion_dim = g_embed_dim
        else:
            self.embed_layer = None
            fusion_dim = g_embed_dim

        self.partial_hazard = PartialHazard(fusion_dim, haz_dim, out_dim, dropout)

    def forward(self, inp, return_hg=False):
        inp_c = inp[:, :self.c_in_dim]
        inp_n = inp[:, self.c_in_dim:]
        inp_n = inp_n.reshape(-1).unsqueeze(1)
        g = dgl.batch([self.g] * inp.shape[0])

        embed = self.embed_layer(inp_c) if self.embed_layer else None

        x = inp_n
        readouts = []
        for i in range(len(self.convs)):
            x = F.relu(self.convs[i](g, x))
            if len(x.shape) == 3: x = torch.mean(x, dim=1) # GAT

            if i == len(self.convs) - 1 and self.pool:
                g, x, _ = self.pool(g, x)

            if self.readout == 'max':
                r = self.max_readout(g, x)
            elif self.readout == 'avg':
                r = self.avg_readout(g, x)
            elif self.readout == 'both':
                r = torch.cat([self.avg_readout(g, x), self.max_readout(g, x)], -1)
            readouts.append(F.relu(r))

        hg = sum(readouts)
        if return_hg: return hg

        if embed is None:
            fusion = hg
        else:
            if self.fusion == 'cat':
                fusion = torch.cat((embed, hg), 1)
            elif self.fusion == 'dot':
                fusion = embed * hg
        out = self.partial_hazard(fusion)
        return out

class D2C:
    def __init__(self, _obj):
        self.__dict__.update(_obj)

class MVP_ori(nn.Module):
    def __init__(self, method, g_in_dim, g_hidden_dim, c_in_dim, c_embed_dim, haz_dim, out_dim, dropout, pool_ratio,
                readout='both', # max, mean
                fusion='cat' # dot
                ):
        super().__init__()
        self.readout = readout
        self.fusion = fusion

        args = D2C({
            'hop': 3,
            'hop_connection': False,
            'lamb': 2.0,
            'patience': 100,
            'sample_neighbor': True,
            'sparse_attention': True,
            'structure_learning': False,
        })

        if method == 'GCN':
            self.conv1 = GCNConv(g_in_dim, g_hidden_dim)
            self.conv2 = GCN(g_hidden_dim, g_hidden_dim)
            self.conv3 = GCN(g_hidden_dim, g_hidden_dim)
        elif method == 'GIN':
            self.lin1 = nn.Linear(g_in_dim, g_hidden_dim)
            self.conv1 = GINConv(self.lin1, train_eps=True)
            self.conv2 = GATConv(g_hidden_dim, g_hidden_dim)
            self.conv3 = GATConv(g_hidden_dim, g_hidden_dim)

        self.pool1 = MVPool(g_hidden_dim, pool_ratio, args)
        self.pool2 = MVPool(g_hidden_dim, pool_ratio, args)
        self.pool3 = MVPool(g_hidden_dim, pool_ratio, args)

        g_embed_dim = g_hidden_dim * (2 if readout == 'both' else 1)

        if c_in_dim:
            self.embed_layer = nn.Sequential(
                nn.Linear(c_in_dim, c_embed_dim, bias=False),
                nn.ReLU()
            )
            if fusion == 'cat':
                fusion_dim = g_embed_dim + c_embed_dim
            else:
                assert g_embed_dim == c_embed_dim
                fusion_dim = g_embed_dim
        else:
            self.embed_layer = None
            fusion_dim = g_embed_dim

        self.partial_hazard = PartialHazard(fusion_dim, haz_dim, out_dim, dropout)

    def forward(self, x, inp_c, edge_index, batch, hook=None, return_hg=False):
        embed = self.embed_layer(inp_c) if self.embed_layer else None

        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, perm, score = self.pool1(x, edge_index, edge_attr, batch)
        if self.readout == 'max':
            x1 = gmp(x, batch)
        elif self.readout == 'avg':
            x1 = gap(x, batch)
        elif self.readout == 'both':
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        if hook: hook(perm, batch, score)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, perm, score = self.pool2(x, edge_index, edge_attr, batch)
        if self.readout == 'max':
            x2 = gmp(x, batch)
        elif self.readout == 'avg':
            x2 = gap(x, batch)
        elif self.readout == 'both':
            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        if hook: hook(perm, batch, score)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, perm, score = self.pool3(x, edge_index, edge_attr, batch)
        if self.readout == 'max':
            x3 = gmp(x, batch)
        elif self.readout == 'avg':
            x3 = gap(x, batch)
        elif self.readout == 'both':
            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        if hook: hook(perm, batch, score)

        hg = F.relu(x1) + F.relu(x2) + F.relu(x3)
        if return_hg: return hg

        if embed is None:
            fusion = hg
        else:
            if self.fusion == 'cat':
                fusion = torch.cat((embed, hg), 1)
            elif self.fusion == 'dot':
                fusion = embed * hg
        out = self.partial_hazard(fusion)
        return out


class MVP(nn.Module):
    def __init__(self, method, inp_dim, graph, g_dims, c_embed_dim, haz_dim, out_dim, dropout, pool_ratio,
                readout='both', # max, mean
                fusion='cat' # dot
                ):
        super().__init__()
        self.readout = readout
        self.fusion = fusion
        self.g = graph
        c_in_dim = inp_dim - (self.g.num_nodes() if self.g else 0)
        self.c_in_dim = c_in_dim

        args = D2C({
            'hop': 3,
            'hop_connection': False,
            'lamb': 2.0,
            'patience': 100,
            'sample_neighbor': True,
            'sparse_attention': True,
            'structure_learning': False,
        })

        if self.g:
            convs = []
            if method == 'GCN':
                convs = [GCNConv(g_dims[0], g_dims[1]),
                        GCN(g_dims[1], g_dims[2]),
                        GCN(g_dims[2], g_dims[3])]
            elif method == 'GIN':
                convs = [GINConv(nn.Linear(g_dims[0], g_dims[1]), train_eps=True),
                        GATConv(g_dims[1], g_dims[2]),
                        GATConv(g_dims[2], g_dims[3])]

            self.convs = nn.ModuleList(convs)
            self.pools = nn.ModuleList([MVPool(d, pool_ratio, args) for d in g_dims[1:]])

            g_embed_dim = g_dims[-1] * (2 if readout == 'both' else 1)
        else:
            g_embed_dim = 0

        if c_in_dim:
            self.embed_layer = nn.Sequential(
                nn.Linear(c_in_dim, c_embed_dim, bias=False),
                nn.ReLU()
            )
            if fusion == 'cat':
                fusion_dim = g_embed_dim + c_embed_dim
            else:
                assert g_embed_dim == c_embed_dim
                fusion_dim = g_embed_dim
        else:
            self.embed_layer = None
            fusion_dim = g_embed_dim

        self.partial_hazard = PartialHazard(fusion_dim, haz_dim, out_dim, dropout)

    def forward(self, inp, hook=None, return_hg=False, return_fusion=False):
        inp_c = inp[:, :self.c_in_dim]
        embed = self.embed_layer(inp_c) if self.embed_layer else None

        if self.g:
            inp_n = inp[:, self.c_in_dim:]
            inp_n = inp_n.reshape(-1).unsqueeze(1)
            g = dgl.batch([self.g] * inp.shape[0])
            edge_index = torch.stack(g.edges())
            batch = torch.tensor([[i] * (inp.shape[1] - self.c_in_dim) for i in range(inp.shape[0])]).flatten().to(g.device)

            edge_attr = None
            x = inp_n
            readouts = []
            for i in range(len(self.convs)):
                x = F.relu(self.convs[i](x, edge_index, edge_attr))
                x, edge_index, edge_attr, batch, perm, score = self.pools[i](x, edge_index, edge_attr, batch)
                if self.readout == 'max':
                    r = gmp(x, batch)
                elif self.readout == 'avg':
                    r = gap(x, batch)
                elif self.readout == 'both':
                    r = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
                if hook: hook(perm, batch, score)
                readouts.append(F.relu(r))

            hg = sum(readouts)
            if return_hg: return hg

        if self.g is None:
            fusion = embed
        elif embed is None:
            fusion = hg
        else:
            if self.fusion == 'cat':
                fusion = torch.cat((embed, hg), 1)
            elif self.fusion == 'dot':
                fusion = embed * hg
        
        if return_fusion: return fusion
        
        out = self.partial_hazard(fusion)
        return out


class DeepSurv(nn.Module):
    def __init__(self, n_in, hidden_layers_sizes=None, activation='rectify', dropout=None, batch_norm=False):
        super().__init__()

        layers = []
        pre_n_layer = n_in
        for n_layer in (hidden_layers_sizes or []):
            layers.append(nn.Linear(pre_n_layer, n_layer))

            if activation == 'rectify':
                layers.append(nn.ReLU())
            elif activation == 'selu':
                layers.append(nn.SELU())
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(n_layer))
            
            if not dropout is None:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(n_layer, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class Nnetsurvival(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden_layers_sizes=None):
        super().__init__()

        layers = []
        pre_n_layer = in_dim
        for n_layer in (hidden_layers_sizes or []):
            layers.append(nn.Linear(pre_n_layer, n_layer))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(n_layer, out_dim))
        # layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
