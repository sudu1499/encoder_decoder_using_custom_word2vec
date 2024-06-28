from torch import nn
import torch
class Encoder(nn.Module):


    def __init__(self,input_size,hidden_size,num_layers):

        super().__init__()

        self.l1=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dtype=torch.float64)

    def forward(self,x,h,c):

        self.lr=self.l1(x,(h,c))

        return self.lr
    

class Decoder(nn.Module):

    def __init__(self,input_size,hidden_size,num_layers,number_classes):

        super().__init__()

        self.l1=nn.LSTM(input_size,number_classes,num_layers,batch_first=True,dtype=torch.float64)

        self.dense=nn.Linear(number_classes,number_classes,dtype=torch.float64)
        self.sm=nn.Softmax(dim=1)

    def forward(self,x,h,c):

        self.r=self.l1(x,(h,c))
        # print(self.r[0].shape)
        return self.sm(self.dense(self.r[0])),self.r[1]
    











