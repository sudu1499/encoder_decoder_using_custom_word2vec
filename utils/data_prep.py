

def data_preparing_to_encoding(data_file,window_size):
    

    #### data_file : path of the file that needs to be word embedded
    #### window_size :size fro preparing the dataset 
    from nltk.text import sent_tokenize
    from nltk.tokenize import word_tokenize
    import numpy as np
    import re
    from utils.continus_bow import c_bag_of_words
    import pickle as pkl
    from utils.one_hot_vocab import one_hot_encode


    with open(data_file,'r',encoding="utf8") as f:
        file=f.read()
    pattern=r"[\"\'\.,\!@#$%^&*()?_\-\{\}:;]"
    data=sent_tokenize(file)
    word_tokenized=[word_tokenize(re.sub(pattern,"",i)) for i in data]


    vocab={}
    count=0
    for i in word_tokenized:
        for j in i:
            try:
                if j not in vocab.keys():
                    vocab[j]=count
                    count+=1
            except:
                print("In the pass")
                pass

    pkl.dump(vocab,open("E:\\encoder_decoder_using_custom_word2vec\\vocabs\\eng_vocab_dict.pkl","wb"))
    ohe_vocab=one_hot_encode(vocab)
    x,y=c_bag_of_words(window_size,ohe_vocab,word_tokenized)
    return x,y