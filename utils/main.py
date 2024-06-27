from utils.data_generation_eng_hindi.generate_data import generate_data_hindi,generate_data_eng

from ed_utils.model import Encoder,Decoder
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
data_path="E:\\encoder_decoder_using_custom_word2vec\\data\\Dataset_English_Hindi.csv"
vocab_path='E:\\encoder_decoder_using_custom_word2vec\\vocabs\\hindi_vocab_dict.pkl'
vocab_ohe_path='E:\\encoder_decoder_using_custom_word2vec\\vocabs\\hindi_vocab_ohe.pkl'
start=300
end=400

x_hindi,d_input_size=generate_data_hindi(data_path,vocab_path,vocab_ohe_path,start,end)

data_path='E:\\encoder_decoder_using_custom_word2vec\\data\\data.txt'
vocab_dict_path="E:\\encoder_decoder_using_custom_word2vec\\vocabs\\eng_vocab_dict.pkl"
wv_path="E:\\encoder_decoder_using_custom_word2vec\\word2vec_model\\model.pkl"

x_eng,e_input_size=generate_data_eng(vocab_dict_path,wv_path,data_path)

##########################################################################

hidden_size=32
num_layers=2

number_classes=d_input_size

enc=Encoder(e_input_size,hidden_size,num_layers).to("cuda")
dec=Decoder(d_input_size,hidden_size,num_layers,number_classes).to("cuda")

loss=nn.CrossEntropyLoss().to("cuda")
opt=Adam(dec.parameters(),lr=0.01)
h,c=torch.rand((num_layers,hidden_size)),torch.rand((num_layers,hidden_size))

for x,y in zip(x_eng,x_hindi):
    x=torch.tensor(np.array(x))
    y=torch.tensor(np.array(y))
    x=x.to("cuda")
    y=y.to("cuda")
    for i in x:
        _,(h,c)=enc(i,h,c)
    ypred=[]
    y_act=[]
    for i in range(len(y)-1):

        op,(h,c)=dec(y[i],(h,c))
        ypred.append(op)
        y_act.append(y[i+1])

    diff=loss(ypred,y_act)

# to be continued