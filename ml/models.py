import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import NNConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import CGConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import AttentiveFP
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_add_pool


class GCN(nn.Module):
    def __init__(self, n_node_feats, dim_out):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 128)
        self.gc1 = GCNConv(in_channels=128, out_channels=128)
        self.gn1 = LayerNorm(128)
        self.gc2 = GCNConv(in_channels=128, out_channels=128)
        self.gn2 = LayerNorm(128)
        self.fc2 = nn.Linear(128, dim_out)

    def forward(self, g, readout=True):
        hx = self.fc1(g.x)
        h = F.relu(self.gn1(self.gc1(hx, g.edge_index)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index)))

        if readout:
            h = global_mean_pool(h, g.batch)

        out = self.fc2(h)

        return out

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            batch = batch.cuda()

            preds = self(batch)
            loss = criterion(batch.y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        self.eval()

        with torch.no_grad():
            preds = [self(batch.cuda()) for batch in data_loader]

        return torch.vstack(preds)

    def emb(self, data_loader):
        embs = list()

        self.eval()
        with torch.no_grad():
            for b in data_loader:
                b = b.cuda()
                hx = self.fc1(b.x)
                h = F.relu(self.gn1(self.gc1(hx, b.edge_index, b.edge_attr)))
                h = F.relu(self.gn2(self.gc2(h, b.edge_index, b.edge_attr)))
                e = global_mean_pool(h, b.batch)
                embs.append(e)

        return torch.vstack(embs)


class GAT(nn.Module):
    def __init__(self, n_node_feats, dim_out):
        super(GAT, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 128)
        self.gc1 = GATConv(in_channels=128, out_channels=128)
        self.gn1 = LayerNorm(128)
        self.gc2 = GATConv(in_channels=128, out_channels=128)
        self.gn2 = LayerNorm(128)
        self.fc2 = nn.Linear(128, dim_out)

    def forward(self, g, readout=True):
        hx = self.fc1(g.x)
        h = F.relu(self.gn1(self.gc1(hx, g.edge_index)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index)))

        if readout:
            h = global_mean_pool(h, g.batch)

        out = self.fc2(h)

        return out

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            batch = batch.cuda()

            preds = self(batch)
            loss = criterion(batch.y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        self.eval()

        with torch.no_grad():
            preds = [self(batch.cuda()) for batch in data_loader]

        return torch.vstack(preds)

    def emb(self, data_loader):
        embs = list()

        self.eval()
        with torch.no_grad():
            for b in data_loader:
                b = b.cuda()
                hx = self.fc1(b.x)
                h = F.relu(self.gn1(self.gc1(hx, b.edge_index, b.edge_attr)))
                h = F.relu(self.gn2(self.gc2(h, b.edge_index, b.edge_attr)))
                e = global_mean_pool(h, b.batch)
                embs.append(e)

        return torch.vstack(embs)


class GIN(nn.Module):
    def __init__(self, n_node_feats, dim_out):
        super(GIN, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 128)
        self.gc1 = GINConv(nn.Linear(128, 128))
        self.gn1 = LayerNorm(128)
        self.gc2 = GINConv(nn.Linear(128, 128))
        self.gn2 = LayerNorm(128)
        self.fc2 = nn.Linear(128, dim_out)

    def forward(self, g, readout=True):
        hx = self.fc1(g.x)
        h = F.relu(self.gn1(self.gc1(hx, g.edge_index)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index)))

        if readout:
            h = global_mean_pool(h, g.batch)

        out = self.fc2(h)

        return out

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            batch = batch.cuda()

            preds = self(batch)
            loss = criterion(batch.y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        self.eval()

        with torch.no_grad():
            preds = [self(batch.cuda()) for batch in data_loader]

        return torch.vstack(preds)

    def emb(self, data_loader):
        embs = list()

        self.eval()
        with torch.no_grad():
            for b in data_loader:
                b = b.cuda()
                hx = self.fc1(b.x)
                h = F.relu(self.gn1(self.gc1(hx, b.edge_index, b.edge_attr)))
                h = F.relu(self.gn2(self.gc2(h, b.edge_index, b.edge_attr)))
                e = global_mean_pool(h, b.batch)
                embs.append(e)

        return torch.vstack(embs)


class ECCNN(nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, dim_out):
        super(ECCNN, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 64)
        self.efc1 = nn.Sequential(nn.Linear(n_edge_feats, 64), nn.ReLU(), nn.Linear(64, 64 * 64))
        self.gc1 = NNConv(64, 64, self.efc1)
        self.gn1 = LayerNorm(64)
        self.efc2 = nn.Sequential(nn.Linear(n_edge_feats, 64), nn.ReLU(), nn.Linear(64, 64 * 64))
        self.gc2 = NNConv(64, 64, self.efc2)
        self.gn2 = LayerNorm(64)
        self.fc2 = nn.Linear(64, dim_out)

    def forward(self, g, readout=True):
        hx = self.fc1(g.x)
        h = F.relu(self.gn1(self.gc1(hx, g.edge_index, g.edge_attr)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index, g.edge_attr)))

        if readout:
            h = global_mean_pool(h, g.batch)

        out = self.fc2(h)

        return out

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            batch = batch.cuda()

            preds = self(batch)
            loss = criterion(batch.y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        self.eval()

        with torch.no_grad():
            preds = [self(batch.cuda()) for batch in data_loader]

        return torch.vstack(preds)

    def emb(self, data_loader):
        embs = list()

        self.eval()
        with torch.no_grad():
            for b in data_loader:
                b = b.cuda()
                hx = self.fc1(b.x)
                h = F.relu(self.gn1(self.gc1(hx, b.edge_index, b.edge_attr)))
                h = F.relu(self.gn2(self.gc2(h, b.edge_index, b.edge_attr)))
                e = global_mean_pool(h, b.batch)
                embs.append(e)

        return torch.vstack(embs)


class CGCNN(nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, dim_out):
        super(CGCNN, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 128)
        self.gc1 = CGConv(128, n_edge_feats)
        self.gn1 = LayerNorm(128)
        self.gc2 = CGConv(128, n_edge_feats)
        self.gn2 = LayerNorm(128)
        self.fc2 = nn.Linear(128, dim_out)

    def forward(self, g, readout=True):
        hx = self.fc1(g.x)
        h = F.relu(self.gn1(self.gc1(hx, g.edge_index, g.edge_attr)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index, g.edge_attr)))

        if readout:
            h = global_mean_pool(h, g.batch)

        out = self.fc2(h)

        return out

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            batch = batch.cuda()

            preds = self(batch)
            loss = criterion(batch.y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        self.eval()

        with torch.no_grad():
            preds = [self(batch.cuda()) for batch in data_loader]

        return torch.vstack(preds)

    def emb(self, data_loader):
        embs = list()

        self.eval()
        with torch.no_grad():
            for b in data_loader:
                b = b.cuda()
                hx = self.fc1(b.x)
                h = F.relu(self.gn1(self.gc1(hx, b.edge_index, b.edge_attr)))
                h = F.relu(self.gn2(self.gc2(h, b.edge_index, b.edge_attr)))
                e = global_mean_pool(h, b.batch)
                embs.append(e)

        return torch.vstack(embs)


class TFGNN(nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, dim_out):
        super(TFGNN, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 128)
        self.gc1 = TransformerConv(in_channels=128, out_channels=128, edge_dim=n_edge_feats)
        self.gn1 = LayerNorm(128)
        self.gc2 = TransformerConv(in_channels=128, out_channels=128, edge_dim=n_edge_feats)
        self.gn2 = LayerNorm(128)
        self.fc2 = nn.Linear(128, dim_out)

    def forward(self, g):
        hx = self.fc1(g.x)
        h = F.relu(self.gn1(self.gc1(hx, g.edge_index, g.edge_attr)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index, g.edge_attr)))
        h = global_mean_pool(h, g.batch)
        out = self.fc2(h)

        return out

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            batch = batch.cuda()

            preds = self(batch)
            loss = criterion(batch.y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        self.eval()

        with torch.no_grad():
            preds = [self(batch.cuda()) for batch in data_loader]

        return torch.vstack(preds)

    def emb(self, data_loader):
        embs = list()

        self.eval()
        with torch.no_grad():
            for b in data_loader:
                b = b.cuda()
                hx = self.fc1(b.x)
                h = F.relu(self.gn1(self.gc1(hx, b.edge_index, b.edge_attr)))
                h = F.relu(self.gn2(self.gc2(h, b.edge_index, b.edge_attr)))
                e = global_mean_pool(h, b.batch)
                embs.append(e)

        return torch.vstack(embs)


class AFP(AttentiveFP):
    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels, num_layers, num_timesteps):
        super(AFP, self).__init__(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_timesteps=num_timesteps
        )

    def fit(self, data_loader, optimizer, criterion):
        train_loss = 0

        self.train()
        for batch in data_loader:
            batch = batch.cuda()

            preds = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(batch.y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        preds = list()

        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.cuda()
                preds.append(self(batch.x, batch.edge_index, batch.edge_attr, batch.batch))

        return torch.vstack(preds)


class CASIN(nn.Module):
    def __init__(self, gnn_react, gnn_env, dim_emb, dim_out):
        super(CASIN, self).__init__()
        self.gnn_react = gnn_react
        self.gnn_env = nn.ModuleList(gnn_env) if isinstance(gnn_env, list) else nn.ModuleList([gnn_env])
        self.fc1 = nn.Linear((len(self.gnn_env) + 1) * dim_emb, 64)
        self.fc2 = nn.Linear(64, dim_out)
        self.attn = nn.Linear((len(self.gnn_env) + 1) * dim_emb, 1)

    def forward(self, g_react, g_env):
        n_atoms = g_react.n_atoms.flatten()
        h_react = self.gnn_react(g_react, readout=False)

        h_env = [self.gnn_env[i](g_env[i]) for i in range(0, len(g_env))]
        attns = torch.zeros((h_react.shape[0], 1)).cuda()
        sum_attns = torch.zeros((h_react.shape[0], 1)).cuda()
        for i in range(0, len(g_env)):
            h_env_r = torch.repeat_interleave(h_env[i], n_atoms, dim=0)
            attns += torch.exp(F.leaky_relu(self.attn(torch.hstack([h_react, h_env_r]))))
            sum_attns += torch.repeat_interleave(global_add_pool(attns, g_react.batch), n_atoms, dim=0)
        norm_attns = attns / sum_attns

        h_react_attn = global_add_pool(norm_attns * h_react, g_react.batch)
        h = torch.hstack([h_react_attn, torch.hstack(h_env)])
        h = F.relu(self.fc1(h))
        out = self.fc2(h)

        return out

    def _emb(self, g_core, g_env):
        h_react = self.gnn_react(g_core, readout=False)
        h_env = torch.hstack([(self.gnn_env[i](g_env[i])) for i in range(0, len(g_env))])

        n_atoms = g_core.n_atoms.flatten()
        h_env_r = torch.repeat_interleave(h_env, n_atoms, dim=0)
        attns = F.relu(self.attn(torch.hstack([h_react, h_env_r])))
        attns = torch.clamp(attns, min=1e-3, max=1e+3)
        sum_attns = torch.repeat_interleave(global_add_pool(attns, g_core.batch), n_atoms, dim=0)
        norm_attns = attns / sum_attns
        h_react_attn = global_add_pool(norm_attns * h_react, g_core.batch)
        e = torch.hstack([h_react_attn, h_env])

        return e

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for g_react, g_env, y in data_loader:
            for g in g_env:
                g.cuda()

            preds = self(g_react.cuda(), g_env)
            loss = criterion(y.cuda(), preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        self.eval()
        preds = list()

        with torch.no_grad():
            for g_react, g_env, y in data_loader:
                for g in g_env:
                    g.cuda()

                preds.append(self(g_react.cuda(), g_env))

        return torch.vstack(preds)

    def emb(self, data_loader):
        embs = list()

        self.eval()
        with torch.no_grad():
            for g_react, g_env, _ in data_loader:
                for g in g_env:
                    g.cuda()

                embs.append(self._emb(g_react.cuda(), g_env))

        return torch.vstack(embs)
