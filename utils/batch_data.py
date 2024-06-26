from torch.utils.data import DataLoader,Dataset
# from utils.data_prep import data_preparing_to_binary_one_hot_encoding
from utils.data_prep import data_preparing_to_encoding
import torch
class my_dataset(Dataset):


    def __init__(self,data_file,window_size):

        self.x,self.y=data_preparing_to_encoding(data_file,window_size)
        self.n=self.x.shape[0]

        self.x=torch.tensor(self.x,dtype=torch.float64)
        self.y=torch.tensor(self.y,dtype=torch.float64)
        
        self.input_size=self.x.shape[-1]
        self.output_size=self.y.shape[-1]

    def __len__(self):
        return self.n
    
    def __getitem__(self,index):

        return self.x[index],self.y[index]
    


def batched_data(data_file,window_size,batch_size):

    from torch.utils.data import DataLoader,Dataset
# from utils.data_prep import data_preparing_to_binary_one_hot_encoding
    from utils.data_prep import data_preparing_to_encoding
    import torch

    dataset=my_dataset(data_file,window_size)

    data=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=1)

    print(f"batched data size is {len(data.dataset)}")
    input_size=dataset.input_size
    output_size=dataset.output_size
    
    return data,input_size,output_size