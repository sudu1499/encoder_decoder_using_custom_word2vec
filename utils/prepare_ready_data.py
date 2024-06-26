import pandas as pd


def create_ready_data(raw_data_path,ready_data_path,start,end):

    data=pd.read_csv(raw_data_path)
    data=data[start:end]['English']
    data.to_csv(ready_data_path,index=None,header=None)
    print(f"ready data creaded in {ready_data_path}")