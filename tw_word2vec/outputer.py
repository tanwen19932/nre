#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : word2vec_zh.py
# @Author: TW
# @Date  : 2018/3/20
# @Desc  :
from tw_word2vec.inputer import SentencesVector
from tw_word2vec.lstm_trainer_zh import LstmTrainer


class Outputer(object):
    def __init__(self, trainer) -> None:
        self.inputer = trainer.inputer
        if not hasattr(trainer, "train"):
            raise Exception("传入文件不包含 train 方法")
        self.trainer = trainer

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

    def getDescription(self,predict_texts: list):
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
        predict_types = self.trainer.predict(SentencesVector(self.inputer,pairs_all=pairs_all,position_all=position_all))
        predict_details = self.inputer.relationWordAdmin.getRelationDetail(pairs_all,position_all,predict_types)
        result = []
        for i in range(len(position_all)):
            entity1 = pairs_all[i][position_all[i][0]]
            entity2 = pairs_all[i][position_all[i][1]]
            obj = {}
            obj["e1"] = entity1.word
            obj["e1_type"] = entity1.flag
            obj["e2"] = entity2.word
            obj["e2_type"] = entity2.flag
            obj["predict_type"] = predict_types[i]
            obj["relation_detail"] = predict_details[i]
            obj["sentence"] = sentences[i]
            result.append(obj)
        return result

