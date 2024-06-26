


#helping function 2 for converting words tokenized in the sentence to binary rep.
def c_bag_of_words(window,vocab_ohe_value,tokenized_word):

    from nltk.text import sent_tokenize
    from nltk.tokenize import word_tokenize
    import numpy as np
    import re    


    x=[]
    y=[]
    for i in tokenized_word:
        temp=[]
        w=window
        middle=int(window/2)
        if len(i)>=window:
            for j in range(len(i)-window+1):
                temp=[]
                temp.append(i[j:w])
                temp=temp[0]
                y.append(temp.pop(middle))
                x.append(temp)
                w+=1
    
    x_o=[]
    y_o=[]
    for i,k in zip(x,y):
        temp=[]
        for j in i:
            temp.append(vocab_ohe_value[j])
        x_o.append(np.array(temp).reshape((1,-1)))
        y_o.append(vocab_ohe_value[k])
    x_o=np.array(x_o)
    y_o=np.array(y_o)

    return x_o,y_o
