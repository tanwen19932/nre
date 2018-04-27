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


class SentencesVector(object):
    sentence_vec = None
    position_vec = None
    pos_vec = None
    classifications_vec = None

    def __init__(self, sentences=None, pairs_all=None, position_all=None, classifications=None) -> None:
        if sentences == None:
            if pairs_all == None or position_all == None:
                raise Exception("传入句子list 或者 pairs_all,position_all")
        else:
            if not isinstance(sentences, list):
                raise Exception("传入句子list")
            pairs_all, position_all = jieba_seg.segListWithNerTag(sentences)
        if len(pairs_all) > MAX_SEQUENCE_LENGTH:
            pairs_all = pairs_all[-MAX_SEQUENCE_LENGTH:]
            position_all = position_all[-MAX_SEQUENCE_LENGTH:]
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
        # from tw_word2vec.cnn_input_zh import all_pos_list
        # all_pos_set = set(all_pos_list)
        # for pairs in pairs_all:
        #     for pair in pairs:
        #         if not all_pos_set.__contains__(pair.flag):
        #             all_pos_list.append(pair.flag)
        #             all_pos_set.add(pair.flag)
        # with open("../data/pos_list.txt", "w") as f:
        #     for pos in all_pos_list:
        #         f.write(pos)
        #         f.write("\n")

        all_pos = list(all_pos_list)
        self.pos_vec = np.zeros((len(pairs_all), MAX_SEQUENCE_LENGTH, len(all_pos)))
        for i in range(len(pairs_all)):
            def getPosIndex(x):
                if all_pos.__contains__(x.flag):
                    return all_pos.index(x.flag)
                else:
                    return 0

            pos_y = list(map(lambda x: getPosIndex(x), pairs_all[i]))
            pos_matrix = to_categorical(pos_y, len(all_pos))
            pos_matrix_all = np.zeros((MAX_SEQUENCE_LENGTH, len(all_pos)))
            pos_matrix_all[-len(pos_matrix):] = pos_matrix
            self.pos_vec[i:] = pos_matrix_all
        if classifications != None:
            classifications_y = list(map(lambda x: types.index(x), classifications))
            self.classifications_vec = to_categorical(classifications_y, len(types))


def getSentenceVectorFromFile(file):
    file_types = []
    file_sentences = []
    with open(file, 'r') as f:
        for line in f.readlines():
            file_types.append(line.split("|")[0])
            file_sentences.append(line.split("|")[1])
    return SentencesVector(file_sentences, file_types)
