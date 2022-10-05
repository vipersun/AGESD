import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.preprocessing import normalize
from utils import load_data, load_graph
from evaluation import compactness,separation,DVI,cluster_prc_f1,translateoutliner
from pre_agesd import AGESD
from kneed import KneeLocator

record_path = 'D://'

def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class Self_AGESD(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(Self_AGESD, self).__init__()
        self.num_clusters = num_clusters
        self.v = v
        # get pretrain model
        self.pre_agesd = AGESD(num_features, hidden_size, embedding_size, alpha)
        # load thr pre-trained GAT model
        self.pre_agesd.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj, M):
        # embedding: Z
        A_pred, z = self.pre_agesd(x, adj, M)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return A_pred, z, q

def agesd(dataset):
    model = Self_AGESD(num_features=args.input_dim, hidden_size=args.hidden1_dim,
                  embedding_size=args.hidden2_dim, alpha=args.alpha, num_clusters=args.n_clusters).to(device)
    # print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Some porcess
    adj, adj_label = load_graph(args.name, args.k)
    adj_dense = adj.to_dense()
    adj_numpy = adj_dense.data.cpu().numpy()
    t = 2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    M = torch.Tensor(M_numpy).cuda()
    adj = adj_dense.cuda()
    adj_label = adj_label.cuda()
    # cluster parameter initiate
    data = torch.Tensor(dataset.x).cuda()
    y = dataset.y
    with torch.no_grad():
        _, z = model.pre_agesd(data, adj, M)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    for epoch in range(args.max_epoch):
        model.train()
        if epoch % args.update_interval == 0:      #[1,3,5]
            # update_interval
            A_pred, z, tmp_q = model(data, adj, M)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res3 = p.data.cpu().numpy().argmax(1)  # P

        A_pred, z, q = model(data, adj, M)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')                                   
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))        
        loss = 10 * kl_loss + re_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # order
    res1 = list(res1)
    n_class = len(set(res1))
    res1[:] = [y - min(res1) for y in res1]     
    while(max(res1) != n_class - 1):      
        for k in range(n_class):
            if k in res1:
                continue
            res1[:] = [y-1 if y >= k else y for y in res1]    
        n_class = len(set(res1))
    return res1,z

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='cite')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=5, type=int)  # [1,3,5]
    parser.add_argument('--hidden1_dim', default=256, type=int)
    parser.add_argument('--hidden2_dim', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.k = None
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    dataset = load_data(args.name)
    args.input_dim = dataset.x.shape[1]
    fun_num = dataset.x.shape[0]
    args.pretrain_path = 'preagesd.pkl'
    args.k = None
    y = dataset.y
    # agesd(dataset)

    # elbow
    x_list = list()
    y_sp_list = list()
    y_cp_list = list()
    for i in range(15):
        if i + 3 > fun_num:
            break
        args.n_clusters = i+3
        print('n_clusters:',args.n_clusters)
        resq, z = agesd(dataset)        
        sp = separation(resq,z)
        cp = compactness(resq,z)
        if sp <= 0 or cp <=0 :
            continue
        if len(list(set(resq))) != args.n_clusters:
            continue
        x_list.append(args.n_clusters)
        y_sp_list.append(sp)
        y_cp_list.append(cp)

    kneedle_cov_cp = KneeLocator(x_list, y_cp_list, curve='convex', direction='increasing', online=False)                 
    kneedle_cov_sp = KneeLocator(x_list, y_sp_list, curve='concave',direction='increasing', online=False)                  
    cp_dis = max(y_cp_list) - min(y_cp_list)    
    sp_dis = max(y_sp_list) - min(y_sp_list)             
    pred_clusters = int(kneedle_cov_cp.elbow*(cp_dis/(cp_dis+sp_dis))+kneedle_cov_sp.elbow*(sp_dis/(cp_dis+sp_dis)))
    args.n_clusters = pred_clusters
    resq, z = agesd(dataset)
    prc, f1 = cluster_prc_f1(y, resq)
    prc_l, f1_l = cluster_prc_f1(y, translateoutliner(list(resq)))
    nmi = nmi_score(y, resq, average_method='arithmetic')

    # record
    record_file_path = record_path + 'output.csv' 
    result_dict = dict()
    result_dict['name'] = args.name
    result_dict['num'] = dataset.x.shape[0]
    result_dict['n_clusters'] = len(set(list(y)))
    result_dict['pred_n_clusters'] = pred_clusters
    result_dict['cp'] = compactness(resq,z)
    result_dict['sp'] = separation(resq,z)
    result_dict['dvi'] = DVI(resq,z)
    result_dict['prc'] = prc
    result_dict['f1'] = f1
    result_dict['prc_del'] = prc_l
    result_dict['f1_del'] = f1_l
    result_dict['nmi'] = nmi
    if not os.path.exists(record_file_path):
        with open(record_file_path, 'a') as new_f:
            csvw = csv.DictWriter(new_f, fieldnames=[k for k in result_dict], lineterminator='\n')
            csvw.writeheader()
        new_f.close()
    with open(record_file_path, 'a') as f:
        csvw = csv.DictWriter(f, fieldnames=[k for k in result_dict], lineterminator='\n')
        csvw.writerows([result_dict])
    f.close()

    # output
    nodes_dict = dict()                                         #{0: 4290240,...}
    with open('nodes//'+args.name+'_nodes.txt','r') as f:
        for line in f.readlines():
            if "\t" in line:
                line_list = line.split('\t')
                nodes_dict[int(line_list[0].replace(' ', ''))] = int(line_list[1].split('\n')[0].replace(' ', ''),16)
        f.close()
    with open(record_path + args.name + '_nodes.txt','w') as f:
        resq = translateoutliner(list(resq))
        function_dict = dict()
        for i in range(len(set(resq))):     
            function_list = list()       
            for j in range(len(resq)):
                if resq[j] == i:
                    function_list.append(hex(nodes_dict[j]))
            function_dict[i] = function_list
        for key,value in function_dict.items():
            print(key,':',value,file=f)
        f.close()

    