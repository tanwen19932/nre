import pandas as pd
from keras.layers import Embedding
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import to_categorical

import tw_word2vec.word2vec as tw_w2v

MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 100

default_model: dict = tw_w2v.get_word2vec_dic("../data/needed_word2vec.bin")
types = ['Component-Whole(e2,e1)', 'Content-Container(e1,e2)', 'Product-Producer(e2,e1)', 'Other', 'Instrument-Agency(e2,e1)', 'Entity-Destination(e1,e2)', 'Entity-Origin(e1,e2)', 'Instrument-Agency(e1,e2)', 'Cause-Effect(e1,e2)', 'Product-Producer(e1,e2)', 'Member-Collection(e1,e2)', 'Message-Topic(e2,e1)', 'Entity-Origin(e2,e1)', 'Component-Whole(e1,e2)', 'Cause-Effect(e2,e1)', 'Content-Container(e2,e1)', 'Member-Collection(e2,e1)', 'Entity-Destination(e2,e1)', 'Message-Topic(e1,e2)']
print("类型个数",len(types))
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)  # 传入我们词向量的字典
tokenizer.fit_on_texts(default_model.keys())  # 传入我们的训练数据，得到训练数据中出现的词的字典

word_index = tokenizer.word_index
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = default_model[word]
    if embedding_vector is not None:
        # 文本数据中的词在词向量字典中没有，向量为取0；如果有则取词向量中该词的向量
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(num_words,  # 词个数
                            EMBEDDING_DIM,  # 300维向量
                            weights=[embedding_matrix],  # 向量矩阵
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


def get_xy(filepath, percent=1):
    # 读取数据
    train = pd.read_csv(filepath_or_buffer=filepath, delimiter='|',
                        # header=["type","e1","e2","doc"],
                        names=["type", "e1", "e2", "doc"]
                        )
    texts = train['doc'].values.tolist()

    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # 限制每篇文章的长度——可作为输入了
    print(data)
    # 打乱文章顺序
    y = train['type'].map(lambda x: types.index(x.replace("\n", '')))
    y = to_categorical(np.asarray(y))  # 转化label
    if (percent == 1):
        return (data, y)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = y[indices]
    num_validation_samples = int((1 - percent) * data.shape[0])
    # 切割数据
    print("drop num_validation_samples ", num_validation_samples)
    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_test = data[-num_validation_samples:]
    y_test = labels[-num_validation_samples:]
    print('Shape of data tensor:', x_train.shape)
    print('Shape of label tensor:', y_train.shape)
    return (x_train, y_train,x_test,y_test)

def get_sentence_vec(list_doc) -> object:
    sequences = tokenizer.texts_to_sequences(list_doc)
    return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_xy("../data/train.txt", 0.8)