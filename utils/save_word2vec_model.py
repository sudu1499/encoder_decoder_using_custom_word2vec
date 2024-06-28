from utils.prepare_ready_data import create_ready_data
from utils.train_model import train_model
import pickle as pkl
import torch
raw_data_path='E:\\encoder_decoder_using_custom_word2vec\\data\\Dataset_English_Hindi.csv'
ready_data_path='E:\\encoder_decoder_using_custom_word2vec\\data\\data.txt'

start=300
end=500

create_ready_data(raw_data_path,ready_data_path,start,end)  ##########################  generated ready data from raw data

window_size=3
batch_size=8
vec_dim=200
epochs=30
model_path="E:\\word_embedding_pytorch_2.0\\model.pkl" 

model=train_model(ready_data_path,window_size,batch_size,vec_dim,epochs)   ##### inside this data also batched and model is trained

torch.save(model,open("E:\\encoder_decoder_using_custom_word2vec\\word2vec_model\\model.pkl","wb"))

print("model is saved in E:\\encoder_decoder_using_custom_word2vec\\word2vec_model\\model.pkl")