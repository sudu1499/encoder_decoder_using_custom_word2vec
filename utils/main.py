from utils.data_generation_eng_hindi.generate_data import generate_data_hindi,generate_data_eng
import torchsummary
from ed_utils.model import Encoder,Decoder
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
from utils.prepare_ready_data import create_ready_data

data_path="E:\\encoder_decoder_using_custom_word2vec\\data\\Dataset_English_Hindi.csv"
vocab_path='E:\\encoder_decoder_using_custom_word2vec\\vocabs\\hindi_vocab_dict.pkl'
vocab_ohe_path='E:\\encoder_decoder_using_custom_word2vec\\vocabs\\hindi_vocab_ohe.pkl'
start=300
end=500
ready_data_path="E:\\encoder_decoder_using_custom_word2vec\\data\\data.txt"
create_ready_data(data_path,ready_data_path,start,end) ########################  ceating ready data from raw data

x_hindi,d_input_size=generate_data_hindi(data_path,vocab_path,vocab_ohe_path,start,end)

data_path='E:\\encoder_decoder_using_custom_word2vec\\data\\data.txt'
vocab_dict_path="E:\\encoder_decoder_using_custom_word2vec\\vocabs\\eng_vocab_dict.pkl"
wv_path="E:\\encoder_decoder_using_custom_word2vec\\word2vec_model\\model.pkl"

x_eng,e_input_size=generate_data_eng(vocab_dict_path,wv_path,data_path)

##########################################################################

##########################################################################
hidden_size=d_input_size
num_layers=2

number_classes=d_input_size

enc=Encoder(e_input_size,hidden_size,num_layers).to("cuda")
dec=Decoder(d_input_size,hidden_size,num_layers,number_classes).to("cuda")

loss=nn.CrossEntropyLoss()
opt=Adam(dec.parameters(),lr=0.001)

h,c=torch.rand((num_layers,hidden_size),dtype=torch.float64,requires_grad=True).to("cuda"),torch.rand((num_layers,hidden_size),dtype=torch.float64,requires_grad=True).to("cuda")

for epoch in range(30):
    for x,y in zip(x_eng,x_hindi):

        x=torch.tensor(np.array(x)).unsqueeze(0)
        y=torch.tensor(np.array(y)).unsqueeze(0)
        x=x.to("cuda")
        y=y.to("cuda")
        for i in x[0]:
            i=i.unsqueeze(0)
            _,(h,c)=enc(i,h,c)
        ypred=[]
        y_act=[]
        for i in range(len(y[0])-1):
            a=y[0][i].unsqueeze(0)
            op,(h,c)=dec(a,h,c)

            ypred.append(op.detach().cpu().numpy())
            y_act.append(y[0][i+1].detach().cpu().numpy())
            # print("ASD",a.shape)
        ypred=np.array(ypred)
        y_act=np.array(y_act)
        ypred=torch.tensor(ypred,dtype=torch.float64,device="cuda",requires_grad=True).squeeze(1)
        y_act=torch.tensor(y_act,dtype=torch.float64,device="cuda",requires_grad=True)
        # print(ypred.shape,"ypred",y_act.shape)
        diff=loss(ypred,y_act)

        opt.zero_grad()
        diff.backward()
        opt.step()
        print("the loss got is ",diff)
# to be continued
