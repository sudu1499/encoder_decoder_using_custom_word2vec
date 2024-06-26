from torch import nn as nn
import torch

class Word2Vec(nn.Module):

    def __init__(self,input_size,output_size,vec_dim):
        
        super().__init__()
        self.l1=nn.Linear(input_size,input_size,dtype=torch.float64)

        self.a1=nn.LeakyReLU()

        self.ll1=nn.Linear(input_size,vec_dim,dtype=torch.float64)

        self.aa1=nn.LeakyReLU()

        self.l2=nn.Linear(vec_dim,output_size,dtype=torch.float64)

        self.a2=nn.Softmax(dim=2)

    def forward(self,x):

        return self.a2(self.l2(self.aa1(self.ll1(self.a1(self.l1(x))))))
    
    def __getitem__(self,index):
        return self.l2.weight[index]
    

