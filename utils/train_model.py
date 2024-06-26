
def train_model(data_file,window_size,batch_size,vec_dim,epochs):

    from utils.model import Word2Vec
    from utils.batch_data import batched_data
    from torch.optim import Adam
    from torch import nn
    data,input_size,output_size=batched_data(data_file,window_size,batch_size)

    model=Word2Vec(input_size,output_size,vec_dim).to("cuda")

    opt=Adam(model.parameters(),lr=0.01)

    loss=nn.CrossEntropyLoss().to("cuda")

    for e in range(epochs):

        total_loss=0
        for x,y in data:

            ypred=model(x.to("cuda"))
            diff=loss(ypred.squeeze().to("cuda"),y.squeeze().to("cuda"))


            opt.zero_grad()
            diff.backward()
            opt.step()

            total_loss+=diff

        print(f"in the epoc {e} loss obtained is {total_loss}")

    return model