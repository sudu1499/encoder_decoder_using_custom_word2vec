import pickle as pkl
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import torch
import re
def generate_data_hindi(data_path,vocab_path,vocab_ohe_path,start,end):
    data=pd.read_csv(data_path)
    data=data['Hindi'][start:end]

    hindi_vocab={}
    c=0
    for i in data:
        for n in i.split():
            hindi_vocab[n]=c
            c+=1
    keys=[[i] for i in hindi_vocab.keys()]
    pkl.dump(hindi_vocab,open(vocab_path,"wb"))
    k=np.array(keys).reshape(-1,1)

    ohe=OneHotEncoder()
    hindi_vocab_ohe=ohe.fit_transform(k).toarray()

    pkl.dump(ohe,open(vocab_ohe_path,"wb"))

    for i,j in zip(hindi_vocab.keys(),hindi_vocab_ohe):
        hindi_vocab[i]=j
    #######################################################
    x_hindi=[]
    for i in data:
        temp=[]
        for j in i.split():
            temp.append(list(hindi_vocab[j]))
        x_hindi.append(temp)
    return x_hindi


def generate_data_eng(vocab_dict_path,wv_path,data_path):

    vocab_dict=pkl.load(open(vocab_dict_path,"rb"))
    w2v=torch.load(open(wv_path,"rb"))

    for i in vocab_dict.keys():
        vocab_dict[i]= w2v[vocab_dict[i]].detach().cpu().numpy()

    data_file=open(data_path,"r").read()
    x_eng=[]
    pattern=r"[\"\'\.,\!@#$%^&*()?_\-\{\}:;]"

    for i in data_file.split("\n"):
        temp=[]
        i=re.sub(pattern,"",i)
        for j in i.split():
            temp.append(vocab_dict[j])
        if len(temp)!=0:
            x_eng.append(temp)
    return x_eng