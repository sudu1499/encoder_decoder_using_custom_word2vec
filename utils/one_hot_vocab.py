from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle as pkl

def one_hot_encode(vocab):

    ohe=OneHotEncoder()

    val=np.array(list(vocab.values())).reshape(-1,1)
    val_ohe=ohe.fit_transform(val).toarray()
    for i,j in zip(vocab.keys(),val_ohe):

        vocab[i]=j
    pkl.dump(vocab,open("E:\\encoder_decoder_using_custom_word2vec\\vocabs\\eng_voacb_ohe.pkl","wb"))
    return  vocab

