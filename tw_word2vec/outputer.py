#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : word2vec_zh.py
# @Author: TW
# @Date  : 2018/3/20
# @Desc  :
from tw_word2vec.inputer import SentencesVector
from tw_word2vec.lstm_trainer_zh import LstmTrainer
import numpy as np


class Outputer(object):
    def __init__(self, trainer) -> None:
        self.inputer = trainer.inputer
        if not hasattr(trainer, "train"):
            raise Exception("传入文件不包含 train 方法")
        self.trainer = trainer

    def getEvaluation(self, testType):
        vector = self.inputer.getSentenceVectorFromFile("../data/test_en.txt")
        classVector = np.eye(vector.classifications_vec.shape)
        print(classVector.shape)
        if testType == "CNN":
            score = self.trainer.model.evaluate(
                {'sequence_input': vector.embedded_sequences, 'posi_input': vector.position_vec,
                 'typeInput': classVector}, vector.classifications_vec, batch_size=128)
        else:
            score = self.trainer.model.evaluate({'sequence_input': vector.embedded_sequences}, vector.classifications_vec, batch_size=128)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def getSentenceRelation(self,predict_texts: list, predict_types):
        relations = []
        for i in range(len(predict_texts)):
            relation_words = self.trainer.relation_word_dic_zh.get(predict_types[i])
            appended = False
            for relation_word in relation_words:
                if predict_texts[i].contains(relation_word):
                    relations.append(relation_word)
                    continue
            if not appended:
                relations.append("未知")
        return relations

    def getDescription(self,predict_texts: list, testType):
        pairs_all = []
        position_all = []
        sentences = []
        for sentence in predict_texts:
            try:
                pairs, position_pair = self.inputer.word_segmentor.segWithNerTag(sentence)
                if len(pairs) <= 100:
                    pairs_all.append(pairs)
                    position_all.append(position_pair)
                    sentences.append(sentence)
            except:
                print(sentence)
                pass
        if testType == "CNN":
            predict_types = self.trainer.predict(SentencesVector(self.inputer, wordPairList_allSen=pairs_all, entityPosition_allSen=position_all))
        else:
            predict_types = self.trainer.predict(SentencesVector(self.inputer, wordPairList_allSen=pairs_all, entityPosition_allSen=position_all))
        predict_details = self.inputer.relationWordAdmin.getRelationDetail(pairs_all, position_all, predict_types)
        result = []
        for i in range(len(position_all)):
            entity1 = pairs_all[i][position_all[i][0]]
            entity2 = pairs_all[i][position_all[i][1]]
            obj = {}
            obj["e1"] = entity1[0]
            obj["e1_type"] = entity1[1]
            obj["e2"] = entity2[0]
            obj["e2_type"] = entity2[1]
            obj["predict_type"] = predict_types[i]
            obj["relation_detail"] = predict_details[i]
            obj["sentence"] = sentences[i]
            result.append(obj)
        return result

