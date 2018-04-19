#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : word2vec_zh.py
# @Author: TW
# @Date  : 2018/3/20
# @Desc  :
import tw_word2vec.lstm_trainer_zh as trainer


# from tw_word2vec.cnn_trainer_zh import *

def getSentenceRelation(predict_texts: list, predict_types):
    relations = []
    for i in range(len(predict_texts)):
        relation_words = trainer.relation_word_dic_zh.get(predict_types[i])
        appended = False
        for relation_word in relation_words:
            if predict_texts[i].contains(relation_word):
                relations.append(relation_word)
                continue
        if not appended:
            relations.append("未知")
    return relations


def getDescription(predict_texts: list):
    from tw_segment.jieba_seg import segListWithNerTag
    pairs_all, position_all = segListWithNerTag(predict_texts)

    predict_types = trainer.predict(trainer.SentencesVector(predict_texts))
    from tw_relation.relations import getRelationDetail
    predict_details = getRelationDetail(predict_texts)
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
        result.append(obj)
    return result


if __name__ == '__main__':
    predict_texts = ["<per>你</per>准备坐<instrument>船</instrument>去那边",
                     "<food>粉丝</food>由<food>马铃薯</food>加工"]
    # print(getSentenceRelation(predict_texts,predict_types))
    import json

    print(json.dumps(getDescription(predict_texts), ensure_ascii=False))

    # get_sentence_vec("<per>你</per>这<per>招<per>打得很不错")
