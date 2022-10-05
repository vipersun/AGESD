import torch
import torch.nn.functional as F
import torch.nn as nn

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.kaiming_uniform_(self.W.data, a=0,mode='fan_out',nonlinearity='leaky_relu') 

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.kaiming_uniform_(self.a_self.data, a=0,mode='fan_out',nonlinearity='leaky_relu') 

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.kaiming_uniform_(self.a_neighs.data, a=0,mode='fan_out',nonlinearity='leaky_relu') 

        self.leakyrelu = nn.LeakyReLU(self.alpha)       

    def forward(self, input, adj, M, concat=True):    # input [N, in_features]
        h = torch.mm(input, self.W)                   # shape [N, out_features] 

        attn_for_self = torch.mm(h,self.a_self)       #(N,1)
        attn_for_neighs = torch.mm(h,self.a_neighs)   #(N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs,0,1)  #(N,N) 
        attn_dense = torch.mul(attn_dense,M)                # [N, N]*[N, N]=>[N, N]
        attn_dense = self.leakyrelu(attn_dense)             #(N,N)

        zero_vec = -9e15*torch.ones_like(adj)               #(N,N)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)                   
        h_prime = torch.matmul(attention,h)                 # N, output_feat

        if concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
            return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        channel = channel
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # AGEMD
        b,c = x.size()
        xt = x.transpose(0,1)               
        xt_3 = torch.unsqueeze(xt,0)        
        y = self.squeeze(xt_3).view(c)      

        min_val = min(i for i in y if i !=0)
        nozero_vec = min_val*torch.ones_like(y) 
        y = torch.where(y > 0, y, nozero_vec)
        return x * y.expand_as(x)

