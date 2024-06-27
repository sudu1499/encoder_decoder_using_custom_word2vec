from torch import nn

class Encoder(nn.Module):


    def __init__(self,input_size,hidden_size,num_layers):

        super().__init__()

        self.l1=nn.LSTM(input_size,hidden_size,num_layers)

    def forward(self,x,h,c):

        self.lr=self.l1(x,(h,c))

        return self.lr[1]
    

class Decoder(nn.Module):

    def __init__(self,input_size,hidden_size,num_layers,number_classes):

        super().__init__()

        self.l1=nn.LSTM(input_size,hidden_size,num_layers)

        self.dense=nn.Linear(input_size,number_classes)
        self.sm=nn.Softmax(dim=0)

    def forward(self,x,h,c):

        self.r=self.l1(x,(h,c))
        
        return self.sm(self.dense(self.r[0])),self.r[1]
    











