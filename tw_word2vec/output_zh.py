#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : word2vec_zh.py
# @Author: TW
# @Date  : 2018/3/20
# @Desc  :
from tw_word2vec.bilstm_attention_trainer_zh import BiLstmAttentionTrainer
from tw_word2vec.bilstm_trainer_zh import BiLstmTrainer
from tw_word2vec.lstm_trainer_zh import LstmTrainer


class OutPuter(object):
    def __init__(self, trainer) -> None:
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
        from tw_segment.jieba_seg import segWithNerTag
        pairs_all = []
        position_all = []
        sentences = []
        for sentence in predict_texts:
            try:
                pairs, position_pair = segWithNerTag(sentence)
                if len(pairs) <= 100:
                    pairs_all.append(pairs)
                    position_all.append(position_pair)
                    sentences.append(sentence)
            except:
                print(sentence)
                pass
        from tw_word2vec.cnn_input_zh import SentencesVector
        predict_types = self.trainer.predict(SentencesVector(pairs_all=pairs_all,position_all=position_all))
        from tw_relation.relations import getRelationDetail
        predict_details = getRelationDetail(sentences)
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


if __name__ == '__main__':
    predict_texts = ["<per>你</per>准备坐<instrument>船</instrument>去那边",
                     "<food>粉丝</food>由<food>马铃薯</food>加工"]
    # print(getSentenceRelation(predict_texts,predict_types))
    outputer = OutPuter(LstmTrainer())
    import json
    print(json.dumps(outputer.getDescription(predict_texts), ensure_ascii=False))

    # get_sentence_vec("<per>你</per>这<per>招<per>打得很不错")
