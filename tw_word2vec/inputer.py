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
from keras.preprocessing.text import Tokenizer


class Configuration(object):
    def __init__(self,
                 word_segmentor=JieBaTokenizer(),
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
        self.word_segmentor = word_segmentor
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
    # 初始化过程中，读取了词向量，位置矩阵，还读取了pos的种类和关系种类及相关词
    def __init__(self, config: Configuration) -> None:
        ##初始化 位置向量矩阵 ：max_sequence_length * 20
        self.config = config
        self.word_segmentor = config.word_segmentor
        self.MAX_SEQUENCE_LENGTH = config.MAX_SEQUENCE_LENGTH
        self.EMBEDDING_DIM = config.EMBEDDING_DIM
        self.MAX_NB_WORDS = config.MAX_NB_WORDS
        if not os.path.isfile(config.position_matrix_file_path):
            position_matrix = np.random.randn(config.MAX_SEQUENCE_LENGTH, 20)
            np.save(config.position_matrix_file_path[0:-4], position_matrix)
        self.position_matrix = np.load(config.position_matrix_file_path)
        print("位置向量矩阵的大小", self.position_matrix.shape)

        ##初始化 tokenizer 转化文本为sequence
        default_model = {}
        default_model = tw_w2v.get_word2vec_dic(config.word2vec_file_path)
        self.tokenizer = Tokenizer(num_words=config.MAX_NB_WORDS)  #
        words = None
        if isinstance(default_model,dict):
            words = default_model.keys()
        else:
            words =  default_model.vocab.keys()

        self.tokenizer.fit_on_texts(words)
        word_index = self.tokenizer.word_index
        self.num_words = min(config.MAX_NB_WORDS, len(word_index)+1)
        ##初始化词向量，词向量矩阵 ： 50000*64
        model_dim =0
        for key in default_model:
            model_dim = default_model[key].shape[0]
            break
        if self.EMBEDDING_DIM != model_dim:
            print("WARN ! 设置的词向量与读取维数不同，默认采用读取的词向量维数。", self.EMBEDDING_DIM, model_dim)
            self.EMBEDDING_DIM =model_dim
            config.EMBEDDING_DIM = model_dim
        self.embedding_matrix = np.zeros((self.num_words, config.EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= config.MAX_NB_WORDS:
                continue
            embedding_vector = default_model[word]
            if embedding_vector is not None:
                # 文本数据中的词在词向量字典中没有，向量为取0；如果有则取词向量中该词的向量
                try:
                    self.embedding_matrix[i] = embedding_vector
                except Exception as e:
                    print(e)
            else:
                print("warn! ",word,"不在词向量列表")
        print("词向量矩阵的大小",self.embedding_matrix.shape)

        ##初始化词性标注List
        self.POS_list = []
        if os.path.exists(config.POS_list_file_path):
            with open(config.POS_list_file_path, encoding="UTF-8") as f:
                for line in f.readlines():
                    if len(line.strip()) > 0:
                        self.POS_list.append(line.strip())
        else:
            file_types = []
            file_sentences = []
            with open(config.corpus_file_path, 'r', encoding="UTF-8") as f:
                for line in f.readlines():
                    file_types.append(line.split("|")[0].strip())
                    file_sentences.append(line.split("|")[1].strip())

            all_pos_set = set(self.POS_list)
            wordPairList_allSen, entityPosition_allSen = self.word_segmentor.segListWithNerTag(file_sentences)
            for pairs in wordPairList_allSen:
                for pair in pairs:
                    if not all_pos_set.__contains__(pair.flag):
                        self.POS_list.append(pair.flag)
                        all_pos_set.add(pair.flag)
            with open(config.POS_list_file_path, "w", encoding="UTF-8") as f:
                for pos in self.POS_list:
                    f.write(pos)
                    f.write("\n")
        print("POS类型", len(self.POS_list))

        ##关系种类，RelationWordAdmin有relations和relation_word_dic
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
        relaTypes = []
        sentences = []
        with open(file, 'r',encoding="UTF-8") as f:
            for line in f.readlines():
                the_type = line.split("|")[0].strip()
                sentence = line.split("|")[1].strip()
                if self.types.__contains__(the_type):
                    relaTypes.append(the_type)
                    sentences.append(sentence)
                else:
                    print("类型不在列表内：",the_type , sentences)
        # 提取出句子及关系分类，传入SentencesVector中
        return SentencesVector(self, sentences, classifications =relaTypes)


class SentencesVector(object):
    sentence_vec = None
    position_vec = None
    pos_vec = None
    classifications_vec = None
    inputer=None

    def __init__(self, inputer: Inputer, sentences=None, wordPairList_allSen=None, entityPosition_allSen=None,
                 classifications=None) -> None:
        self.inputer = inputer
        config = inputer.config
        classifications_all = None
        if sentences == None:
            if wordPairList_allSen == None or entityPosition_allSen == None:
                raise Exception("传入句子list 或者 wordPairList_allSen,entityPosition_allSen")
        else:
            if not isinstance(sentences, list):
                raise Exception("传入句子list")
            wordPairList_allSen = []
            entityPosition_allSen = []
            if not classifications is None:
                classifications_all = []
            for i in range(len(sentences)):
                sentence = sentences[i]
                try:
                    # pairList_wordPos是一个句子中的词与词性构成的元组, position_entityPair是一个句子中的两个实体的位置的元组
                    pairList_wordPos, position_entityPair = inputer.word_segmentor.segWithNerTag(sentence)
                    wordPairList_allSen.append(pairList_wordPos)
                    entityPosition_allSen.append(position_entityPair)
                    if not classifications_all is None:
                        classifications_all.append(classifications[i])
                except Exception as ex:
                    print("[句子错误!]-", sentence)
                    pass

        # if len(wordPairList_allSen) > config.MAX_SEQUENCE_LENGTH:
        #     wordPairList_allSen = wordPairList_allSen[-config.MAX_SEQUENCE_LENGTH:]
        #     entityPosition_allSen = entityPosition_allSen[-config.MAX_SEQUENCE_LENGTH:]
        # 获取句子向量
        #     获取每个句子（去掉标点）
        texts = list(map(lambda pair: reduce(lambda x, y: x + y, map(lambda x: x[0] + " ", pair)), wordPairList_allSen))
        sequences = inputer.tokenizer.texts_to_sequences(texts)
        self.sentence_vec = pad_sequences(sequences, maxlen=config.MAX_SEQUENCE_LENGTH)

        # 获取位置向量
        # 三维矩阵，每个句子对应一个二维矩阵，该二维矩阵是设计规则生成出来的，跟句子中的每个词都有关系
        self.position_vec = np.zeros((len(entityPosition_allSen), config.MAX_SEQUENCE_LENGTH, inputer.position_matrix.shape[1]*2))
        for i in range(len(entityPosition_allSen)):
            e1_position = entityPosition_allSen[i][0]
            e2_position = entityPosition_allSen[i][1]
            sentence_posi_maxtrix = np.zeros((config.MAX_SEQUENCE_LENGTH, inputer.position_matrix.shape[1]*2))
            tokens = list(map(lambda x: x[0], wordPairList_allSen[i]))
            for j in range(len(tokens)):
                e1_pv = inputer.position_matrix[j - e1_position]
                e2_pv = inputer.position_matrix[j - e2_position]
                word_position_matrix = np.append(e1_pv, e2_pv)
                sentence_posi_maxtrix[-len(tokens) + j] = word_position_matrix
            self.position_vec[i:] = sentence_posi_maxtrix

        # 获取词性向量
        # from tw_word2vec.cnn_input_zh import all_pos_list
        # all_pos_set = set(all_pos_list)
        # for pairs in wordPairList_allSen:
        #     for pair in pairs:
        #         if not all_pos_set.__contains__(pair.flag):
        #             all_pos_list.append(pair.flag)
        #             all_pos_set.add(pair.flag)
        # with open("../data/pos_list.txt", "w") as f:
        #     for pos in all_pos_list:
        #         f.write(pos)
        #         f.write("\n")

        # 获取词性向量，先把词、跟词性的元组对的词性部分换成词性在all_pos中的索引，再把索引变成one-hot向量
        # 三维矩阵，每个句子的每个词都有一个one-hot向量表示词性
        all_pos = list(inputer.POS_list)
        print("all_pos:" + str(len(all_pos)))
        self.pos_vec = np.zeros((len(wordPairList_allSen), config.MAX_SEQUENCE_LENGTH, len(all_pos)))
        for i in range(len(wordPairList_allSen)):
            def getPosIndex(x):
                if all_pos.__contains__(x[1]):
                    return all_pos.index(x[1])
                else:
                    return 0

            pos_y = list(map(lambda x: getPosIndex(x), wordPairList_allSen[i]))
            # 转换成one-hot矩阵
            pos_matrix = to_categorical(pos_y, len(all_pos))
            pos_matrix_all = np.zeros((config.MAX_SEQUENCE_LENGTH, len(all_pos)))
            pos_matrix_all[-len(pos_matrix):] = pos_matrix
            self.pos_vec[i:] = pos_matrix_all
        # 这是relation的分类矩阵
        # 跟对pos，posi的处理一样，最后得到的是一个二维矩阵，每个句子对应一个One-hot向量
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
