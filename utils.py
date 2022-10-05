import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
import networkx as nx

def load_graph(dataset, k):
    if k:
        path = 'graph/{}{}_graph.txt'.format(dataset, k) 
    else:
        path = 'graph/{}_graph.txt'.format(dataset) 

    data = np.loadtxt('data/{}.txt'.format(dataset))
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)                       
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),        
                     dtype=np.int32).reshape(edges_unordered.shape)

    # DAEGC
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(n, n), dtype=np.float32)

    # AGEMD
    nodes_list = [i for i in range(n)]
    edge_list = list()
    row = np.array([])
    col = np.array([])
    weight = np.array([])
    for edge in edges:
        line = list()
        row = np.append(row, edge[0])
        col = np.append(col, edge[1])
        line= list(edge)
        line.append(1)
        edge_list.append(tuple(line))
    G = nx.DiGraph() 
    G.add_nodes_from(nodes_list)
    G.add_weighted_edges_from(edge_list)
    nodes_dict = dict()
    for node in nodes_list:
        nodes_dict[node] = float(format(nx.closeness_centrality(G,node),"0.6f"))
    nodes_dict_list = sorted(nodes_dict.items(), key=lambda x:x[0])
    nodes_data = dict()
    for value in nodes_dict_list:
        nodes_data[list(value)[0]] = list(value)[1]
    max_val = max(nodes_data.values())
    min_val = min(nodes_data.values())
    for key in nodes_data.keys():               
        nodes_data[key] = float(format((nodes_data[key])/(max_val - min_val),"0.6f"))
    for i in range(len(col)):
        c = nodes_data[col[i]]
        r = nodes_data[row[i]]
        weight = np.append(weight,r)#r*cr
    adj = sp.coo_matrix((weight, (row, col)), shape=(n, n))
    adj_noeye = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj_noeye + sp.eye(adj_noeye.shape[0])                                
    adj = normalize(adj)                                                        
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_label = sparse_mx_to_torch_sparse_tensor(adj_noeye)
    return adj,adj_label


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class load_data(Dataset):
    def __init__(self, dataset):
        self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
        self.y = np.loadtxt('label/{}_label.txt'.format(dataset), dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))


