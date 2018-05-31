#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : inputer.py
# @Author: TW
# @Date  : 2018/5/17
# @Desc  : inputer 包含了读取关系分类体系，读取语料，词向量,位置向量，词性向量
import os
from functools import reduce

import numpy as np
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import tw_word2vec.word2vec as tw_w2v
from tw_relation.relation_admin import RelationWordAdmin, JieBaTokenizer
from tw_segment import jieba_seg
from keras.preprocessing.text import Tokenizer


class Configuration(object):
    def __init__(self,
                 tokenizer=JieBaTokenizer(),
                 MAX_NB_WORDS=50000,
                 EMBEDDING_DIM=64,
                 MAX_SEQUENCE_LENGTH=100,
                 position_matrix_file_path="../data/posi_matrix.npy",
                 word2vec_file_path="../data/needed_zh_word2vec.bin",
                 POS_list_file_path="../data/pos_list.txt",
                 types_file_path="../data/relations_zh.txt",
                 corpus_file_path="../data/train_zh.txt",
                 model_file_path="../data/model/re_zh_model.lstm.hdf5",
                 ) -> None:
        self.tokenizer = tokenizer
        self.MAX_NB_WORDS = MAX_NB_WORDS
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.position_matrix_file_path = position_matrix_file_path
        self.word2vec_file_path = word2vec_file_path
        self.POS_list_file_path = POS_list_file_path
        self.types_file_path = types_file_path
        self.corpus_file_path = corpus_file_path
        self.model_file_path = model_file_path


class Inputer(object):
    def __init__(self, config: Configuration) -> None:
        ##初始化 位置向量 20维数
        self.config = config
        self.MAX_SEQUENCE_LENGTH = config.MAX_SEQUENCE_LENGTH
        self.EMBEDDING_DIM = config.EMBEDDING_DIM
        self.MAX_NB_WORDS = config.MAX_NB_WORDS
        if not os.path.isfile(config.position_matrix_file_path):
            position_matrix = np.random.randn(config.MAX_SEQUENCE_LENGTH, 20)
            np.save(config.position_matrix_file_path[0:-4], position_matrix)
        self.position_matrix = np.load(config.position_matrix_file_path)
        print("位置向量", self.position_matrix.shape)

        ##初始化 tokenizer 转化文本为sequence
        default_model: dict = tw_w2v.get_word2vec_dic(config.word2vec_file_path)
        self.tokenizer = Tokenizer(num_words=config.MAX_NB_WORDS)  # 传入我们词向量的字典
        self.tokenizer.fit_on_texts(default_model.keys())  # 传入我们的训练数据，得到训练数据中出现的词的字典
        word_index = self.tokenizer.word_index
        self.num_words = min(config.MAX_NB_WORDS, len(word_index))

        ##初始化词向量
        self.embedding_matrix = np.zeros((self.num_words, config.EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= config.MAX_NB_WORDS:
                continue
            embedding_vector = default_model[word]
            if embedding_vector is not None:
                # 文本数据中的词在词向量字典中没有，向量为取0；如果有则取词向量中该词的向量
                try:
                    self.embedding_matrix[i] = embedding_vector
                except:
                    pass

        print("词向量",self.embedding_matrix.shape)
        ##初始化词性标注List
        self.POS_list = []
        if os.path.exists(config.POS_list_file_path):
            with open(config.POS_list_file_path) as f:
                for line in f.readlines():
                    if len(line.strip()) > 0:
                        self.POS_list.append(line.strip())
        else:
            file_types = []
            file_sentences = []
            with open(config.corpus_file_path, 'r') as f:
                for line in f.readlines():
                    file_types.append(line.split("|")[0].strip())
                    file_sentences.append(line.split("|")[1].strip())

            all_pos_set = set(self.POS_list)
            pairs_all, position_all = jieba_seg.segListWithNerTag(file_sentences)
            for pairs in pairs_all:
                for pair in pairs:
                    if not all_pos_set.__contains__(pair.flag):
                        self.POS_list.append(pair.flag)
                        all_pos_set.add(pair.flag)
            with open(config.POS_list_file_path, "w") as f:
                for pos in self.POS_list:
                    f.write(pos)
                    f.write("\n")
                    ##初始化分类类型
        print("POS类型", len(self.POS_list))
        self.relationWordAdmin = RelationWordAdmin(config.types_file_path);
        self.types = self.relationWordAdmin.relations
        print("分类类型", len(self.types))


    def getWordEmbedding(self):
        embedding_layer = Embedding(self.num_words,  # 词个数
                                    self.config.EMBEDDING_DIM,  # 维向量
                                    weights=[self.embedding_matrix],  # 向量矩阵
                                    input_length=self.MAX_SEQUENCE_LENGTH,
                                    trainable=False)
        return embedding_layer

    def getSentenceVectorFromFile(self, file):
        file_types = []
        file_sentences = []
        with open(file, 'r') as f:
            for line in f.readlines():
                the_type = line.split("|")[0].strip()
                sentence = line.split("|")[1].strip()
                if self.types.__contains__(the_type):
                    file_types.append(the_type)
                    file_sentences.append(sentence)
                else:
                    print("类型不在列表内：",the_type , file_sentences)
        return SentencesVector(self, file_sentences, classifications =file_types)


class SentencesVector(object):
    sentence_vec = None
    position_vec = None
    pos_vec = None
    classifications_vec = None
    inputer=None

    def __init__(self, inputer: Inputer, sentences=None, pairs_all=None, position_all=None,
                 classifications=None) -> None:
        self.inputer = inputer
        config = inputer.config
        classifications_all = None
        if sentences == None:
            if pairs_all == None or position_all == None:
                raise Exception("传入句子list 或者 pairs_all,position_all")
        else:
            if not isinstance(sentences, list):
                raise Exception("传入句子list")
            pairs_all = []
            position_all = []
            if not classifications is None:
                classifications_all = []
            for i in range(len(sentences)):
                sentence = sentences[i]
                try:
                    pairs, position_pair = jieba_seg.segWithNerTag(sentence)
                    pairs_all.append(pairs)
                    position_all.append(position_pair)
                    if not classifications_all is None:
                        classifications_all.append(classifications[i])
                except Exception as ex:
                    print("[句子错误!]-", sentence)
                    pass

        # if len(pairs_all) > config.MAX_SEQUENCE_LENGTH:
        #     pairs_all = pairs_all[-config.MAX_SEQUENCE_LENGTH:]
        #     position_all = position_all[-config.MAX_SEQUENCE_LENGTH:]
        # 获取句子向量
        texts = list(map(lambda pair: reduce(lambda x, y: x + y, map(lambda x: x.word + " ", pair)), pairs_all))
        sequences = inputer.tokenizer.texts_to_sequences(texts)
        self.sentence_vec = pad_sequences(sequences, maxlen=config.MAX_SEQUENCE_LENGTH)
        # 获取位置向量
        self.position_vec = np.zeros((len(position_all), config.MAX_SEQUENCE_LENGTH, inputer.position_matrix.shape[1]*2))
        for i in range(len(position_all)):
            e1_position = position_all[i][0]
            e2_position = position_all[i][1]
            sentence_posi_maxtrix = np.zeros((config.MAX_SEQUENCE_LENGTH, inputer.position_matrix.shape[1]*2))
            tokens = list(map(lambda x: x.word, pairs_all[i]))
            for j in range(len(tokens)):
                e1_pv = inputer.position_matrix[j - e1_position]
                e2_pv = inputer.position_matrix[j - e2_position]
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
        all_pos = list(inputer.POS_list)
        self.pos_vec = np.zeros((len(pairs_all), config.MAX_SEQUENCE_LENGTH, len(all_pos)))
        for i in range(len(pairs_all)):
            def getPosIndex(x):
                if all_pos.__contains__(x.flag):
                    return all_pos.index(x.flag)
                else:
                    return 0

            pos_y = list(map(lambda x: getPosIndex(x), pairs_all[i]))
            pos_matrix = to_categorical(pos_y, len(all_pos))
            pos_matrix_all = np.zeros((config.MAX_SEQUENCE_LENGTH, len(all_pos)))
            pos_matrix_all[-len(pos_matrix):] = pos_matrix
            self.pos_vec[i:] = pos_matrix_all
        if classifications_all != None:
            classifications_y = list(map(lambda x: inputer.types.index(x), classifications_all))
            self.classifications_vec = to_categorical(classifications_y, len(inputer.types))

    def prop2index(self, prop):
        output = []
        for row in prop:
            max_index = row.argsort()[-1]
            # raw_type = self.inputer.types[y_test[i].argsort()[-1]]
            # for i in range(len(row)):
            #     print(self.inputer.types[i], row[i])

            predict_type = self.inputer.types[max_index]
            if row[max_index]<0.5:
                print("最大概率为",row[max_index])
                predict_type = "无"
            output.append(predict_type)
        return output
