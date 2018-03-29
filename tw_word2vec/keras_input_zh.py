#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : word2vec_zh.py
# @Author: TW
# @Date  : 2018/3/20
# @Desc  :
import json
import os
from functools import reduce

import numpy as np
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import tw_word2vec.word2vec as tw_w2v
from tw_segment import jieba_seg

MAX_NB_WORDS = 50000
EMBEDDING_DIM = 64
MAX_SEQUENCE_LENGTH = 100

default_model: dict = tw_w2v.get_word2vec_dic("../data/needed_zh_word2vec.bin")
from tw_relation.relations import relations_zh, relation_word_dic_zh, getRelationDetail

types = relations_zh
print("类型个数", len(types))
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)  # 传入我们词向量的字典
tokenizer.fit_on_texts(default_model.keys())  # 传入我们的训练数据，得到训练数据中出现的词的字典
word_index = tokenizer.word_index
num_words = min(MAX_NB_WORDS, len(word_index))
all_pos_list = []
with open("../data/pos_list.txt") as f:
    for line in f.readlines():
        if len(line.strip()) > 0:
            all_pos_list.append(line.strip())
##获取embedding的矩阵。主要是kears的矩阵index转化
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = default_model[word]
    if embedding_vector is not None:
        # 文本数据中的词在词向量字典中没有，向量为取0；如果有则取词向量中该词的向量
        try:
            embedding_matrix[i] = embedding_vector
        except:
            pass

embedding_layer = Embedding(num_words,  # 词个数
                            EMBEDDING_DIM,  # 维向量
                            weights=[embedding_matrix],  # 向量矩阵
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
##初始化位置向量
if not os.path.isfile("../data/posi_matrix.npy"):
    position_matrix = np.random.randn(100, 20)
    np.save("../data/posi_matrix", position_matrix)
position_matrix = np.load("../data/posi_matrix.npy")
keyword = {}

with open("../data/tf_idf.txt", 'r') as load_f:
    keyword = json.load(load_f)


class SentencesVector():
    sentence_vec = None
    position_vec = None
    pos_vec = None
    classifications_vec = None

    def __init__(self, sentences, classifications=None) -> None:
        if not isinstance(sentences, list):
            raise Exception("传入句子list")
        pairs_all,position_all = jieba_seg.segListWithNerTag(sentences)
        # 获取句子向量
        texts = list(map(lambda pair: reduce(lambda x, y: x + y, map(lambda x: x.word + " ", pair)), pairs_all))
        sequences = tokenizer.texts_to_sequences(texts)
        self.sentence_vec = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        # 获取位置向量
        self.position_vec = np.zeros((len(position_all), MAX_SEQUENCE_LENGTH, 40))
        for i in range(len(position_all)):
            e1_position = position_all[i][0]
            e2_position = position_all[i][1]
            sentence_posi_maxtrix = np.zeros((MAX_SEQUENCE_LENGTH, 40))
            tokens = list(map(lambda x: x.word, pairs_all[i]))
            for j in range(len(tokens)):
                e1_pv = position_matrix[j - e1_position]
                e2_pv = position_matrix[j - e2_position]
                word_position_matrix = np.append(e1_pv, e2_pv)
                sentence_posi_maxtrix[-len(tokens) + j] = word_position_matrix
            self.position_vec[i:] = sentence_posi_maxtrix
        # 获取词性向量
        from tw_word2vec.keras_input_zh import all_pos_list
        all_pos_set = set(all_pos_list)
        for pairs in pairs_all:
            for pair in pairs:
                if not all_pos_set.__contains__(pair.flag):
                    all_pos_list.append(pair.flag)
                    all_pos_set.add(pair.flag)
        with open("../data/pos_list.txt", "w") as f:
            for pos in all_pos_list:
                f.write(pos)
                f.write("\n")

        all_pos = list(all_pos_list)
        self.pos_vec = np.zeros((len(pairs_all), MAX_SEQUENCE_LENGTH, len(all_pos)))
        for i in range(len(pairs_all)):
            pos_y = list(map(lambda x: all_pos.index(x.flag), pairs_all[i]))
            pos_matrix = to_categorical(pos_y, len(all_pos))
            pos_matrix_all = np.zeros((MAX_SEQUENCE_LENGTH, len(all_pos)))
            pos_matrix_all[-len(pos_matrix):] = pos_matrix
            self.pos_vec[i:] = pos_matrix_all
        if classifications != None:
            classifications_y = list(map(lambda x: types.index(x), classifications))
            self.classifications_vec = to_categorical(classifications_y, len(types))


import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import MaxPooling1D, Dropout, regularizers
from keras.layers import Dense, Input, Flatten
from keras.models import Model, load_model
import numpy as np

from tw_keras.multi_layer import MultiConv1D


def train(sentences_vector: SentencesVector):
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name="sequence_input")  # 100*1最多100个词组成输入
    embedded_sequences = embedding_layer(sequence_input)  # 句子转为向量矩阵 训练集大小*100*64维
    # model test2
    posi_input = Input(shape=(MAX_SEQUENCE_LENGTH, 40), name="posi_input")
    pos_input = Input(shape=(MAX_SEQUENCE_LENGTH, len(all_pos_list)), name="pos_input")
    embedded_sequences = keras.layers.concatenate([embedded_sequences, posi_input, pos_input])
    # conv1d_1s = MultiConv1D(filters=[90, 80, 70, 50, 30, 10], kernel_size=[3, 4, 5], activation='relu')
    conv1d_1s = MultiConv1D(filters=[10], kernel_size=[3], activation='relu')
    best_model = None
    count = 0
    for conv1d in conv1d_1s:
        c1 = conv1d(embedded_sequences)
        c1 = MaxPooling1D(pool_size=3)(c1)
        c1 = Dropout(rate=0.7)(c1)
        c1 = Flatten()(c1)
        # c1 = Dense(128, activation='relu')(c1)  # 128全连接
        # c1 = Dense(64, activation='relu')(c1)  # 64全连接
        preds = Dense(len(types), activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                      activity_regularizer=regularizers.l1(0.001))(c1)  # softmax分类
        model = Model(inputs=[sequence_input, posi_input, pos_input], outputs=preds)
        print(model.summary())
        adam = optimizers.Adam(lr=0.001, decay=0.0001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=["categorical_accuracy"])

        # 如果希望短一些时间可以，epochs调小

        # ModelCheckpoint回调函数将在每个epoch后保存模型到filepath，当save_best_only=True保存验证集误差最小的参数
        file_path = "../data/model/re_zh_model.temp" + str(count) + ".hdf5"
        checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        # 当监测值不再改善时，该回调函数将中止训练
        early = EarlyStopping(monitor="val_loss", mode="min", patience=50)

        # 开始训练
        callbacks_list = [checkpoint, early]  # early
        # And trained it via:
        model.fit({'sequence_input': sentences_vector.sentence_vec, 'posi_input': sentences_vector.position_vec,
                   'pos_input': sentences_vector.pos_vec},
                  sentences_vector.classifications_vec,
                  batch_size=128,
                  epochs=500,
                  # validation_split=0.2,
                  # validation_data=({'sequence_input': x_test, 'posi_input': x_test_posi}, y_test),
                  callbacks=callbacks_list)
        count += 1
        best_model = model
    return best_model


def predict(sentence_vector: SentencesVector):
    id = model.predict({'sequence_input': sentence_vector.sentence_vec, 'posi_input': sentence_vector.position_vec,
                        'pos_input': sentence_vector.pos_vec})
    output = []
    for row in id:
        max_index = row.argsort()[-1]
        # raw_type = types[y_test[i].argsort()[-1]]
        for i in range(len(row)):
            print(types[i],row[i])
        predict_type = types[max_index]
        output.append(predict_type)
    return output


def getSentenceVectorFromFile(file):
    file_types = []
    file_sentences = []
    with open(file, 'r') as f:
        for line in f.readlines():
            file_types.append(line.split("|")[0])
            file_sentences.append(line.split("|")[1])
    return SentencesVector(file_sentences, file_types)


def getSentenceRelation(predict_texts:list,predict_types):
    relations = []
    for i in range(len(predict_texts)):
        relation_words = relation_word_dic_zh.get(predict_types[i])
        appended = False
        for relation_word in relation_words:
            if predict_texts[i].contains(relation_word):
                relations.append(relation_word)
                continue
        if not appended:
            relations.append("未知")
    return relations

if __name__ == '__main__':
    import os
    model_path = "../data/model/re_zh_model.temp0.hdf5"
    if not os.path.exists(model_path):
        vector = getSentenceVectorFromFile("../data/train_zh.txt")
        print(vector.sentence_vec)
        print(vector.position_vec)
        print(vector.pos_vec)
        print(vector.classifications_vec)
        model = train(vector)
        model.save(model_path)
    model = load_model(model_path)
    predict_texts = ["<per>你</per>准备坐<instrument>船</instrument>去那边",
                                      "<food>粉丝</food>由<food>马铃薯</food>加工"]
    predict_types = predict(SentencesVector(predict_texts))
    print(predict_types)
    # print(getSentenceRelation(predict_texts,predict_types))
    print(getRelationDetail(predict_texts))


    # get_sentence_vec("<per>你</per>这<per>招<per>打得很不错")
