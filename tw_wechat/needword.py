import gensim
import os
import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
# 对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示，超过max_features的单词被丢掉
# 使用一系列文档来生成token词典，texts为list类，每个元素为一个文档
train = pd.read_csv(filepath_or_buffer="../data/train.txt",delimiter='|',
                    # header=["type","e1","e2","doc"],
                    names=["type","e1","e2","doc"]
                    )
test = pd.read_csv(filepath_or_buffer="../data/test.txt",delimiter='|')

print(train.columns)
list_sentences_train = train['doc'].fillna("CVxTz").values
train_type = train['type'].values
list_sentences_test = train['doc'].fillna("CVxTz").values


need_word2vec = {}
tokenizer = text.Tokenizer(num_words=100)
tokenizer.fit_on_texts(list(list_sentences_train))
word_vec = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), '../data/word2vec/GoogleNews-vectors-negative300.bin'), binary=True)

i = 0
for word in tokenizer.word_index:
    if(word_vec.__contains__(word)):
        need_word2vec[i]=word_vec[word]
    else:need_word2vec[i]=np.zeros((1,300))
    i+=1
del word_vec
import pickle
f = open('needed_word2vec.bin', 'wb')
f.write(pickle.dumps(need_word2vec))
f.close()


list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

x_train = sequence.pad_sequences(list_tokenized_train, maxlen=100)
x_test = sequence.pad_sequences(list_tokenized_test, maxlen=100)



