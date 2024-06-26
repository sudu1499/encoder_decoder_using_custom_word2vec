from utils.data_generation_eng_hindi.generate_data import generate_data_hindi,generate_data_eng



data_path="E:\\encoder_decoder_using_custom_word2vec\\data\\Dataset_English_Hindi.csv"
vocab_path='E:\\encoder_decoder_using_custom_word2vec\\vocabs\\hindi_vocab_dict.pkl'
vocab_ohe_path='E:\\encoder_decoder_using_custom_word2vec\\vocabs\\hindi_vocab_ohe.pkl'
start=300
end=400

x_hindi=generate_data_hindi(data_path,vocab_ohe_path,vocab_ohe_path,start,end)

data_path='E:\\encoder_decoder_using_custom_word2vec\\data\\data.txt'
vocab_dict_path="E:\\encoder_decoder_using_custom_word2vec\\vocabs\\eng_vocab_dict.pkl"
wv_path="E:\\encoder_decoder_using_custom_word2vec\\word2vec_model\\model.pkl"

x_eng=generate_data_eng(vocab_dict_path,wv_path,data_path)