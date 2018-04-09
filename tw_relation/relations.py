#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : relation.py
# @Author: TW
# @Date  : 2018/3/27
# @Desc  :
from pprint import pprint

from tw_segment.jieba_seg import *

relations_en = ['Component-Whole(e1,e2)',
                'Component-Whole(e2,e1)',
                'Content-Container(e1,e2)',
                'Content-Container(e2,e1)',
                'Product-Producer(e1,e2)',
                'Product-Producer(e2,e1)',
                'Instrument-Agency(e1,e2)',
                'Instrument-Agency(e2,e1)',
                'Entity-Destination(e1,e2)',
                'Entity-Destination(e2,e1)',
                'Entity-Origin(e1,e2)',
                'Entity-Origin(e2,e1)',
                'Cause-Effect(e1,e2)',
                'Cause-Effect(e2,e1)',
                'Member-Collection(e1,e2)',
                'Member-Collection(e2,e1)',
                'Message-Topic(e1,e2)',
                'Message-Topic(e2,e1)',
                'Other']
relations_zh = ['部分-整体(e1,e2)',
                '部分-整体(e2,e1)',
                '内容-容器(e1,e2)',
                '内容-容器(e2,e1)',
                '产品-生产者(e1,e2)',
                '产品-生产者(e2,e1)',
                '成员-组织(e1,e2)',
                '成员-组织(e2,e1)',
                '工具-代理(e1,e2)',
                '工具-代理(e2,e1)',
                '起因-影响(e1,e2)',
                '起因-影响(e2,e1)',
                '消息-话题(e1,e2)',
                '消息-话题(e2,e1)',
                '无']
relation_word_dic_zh = {}


def getRelationWord(relation):
    result = []
    try:
        with open("../data/relation/" + relation + ".txt", "r") as f:
            for line in f.readlines():
                result.append(line)
    except:
        pass
    return result


for relation_zh in relations_zh:
    relation_word_dic_zh[relation_zh] = getRelationWord(relation_zh)

from pyhanlp import *


def generateRelationWord():
    # 首先进行读取文件 后进行hdp聚类
    sentence_list = []
    with open("../data/train_zh.txt", "r") as f:
        for line in f.readlines():
            sentence_list.append(line.split("|")[1].strip())
    detail_relatino = getRelationDetail(sentence_list)
    print(detail_relatino)


def getRelationDetail(sentence_list):
    relations_detail = []
    pairs_all, position_all = segListWithNerTag(sentence_list)
    # 聚类获取结果
    # corpus = []
    # words_list = list(map(lambda pairs: map(lambda x: x.word, pairs), pairs_all))
    # from gensim import corpora
    # dictionary = corpora.Dictionary(words_list)
    # for words in words_list:
    #     corpus.append(dictionary.doc2bow(words))
    # from gensim.models import HdpModel
    # hdp = HdpModel(corpus, dictionary)
    # a = hdp.print_topics(num_topics=2, num_words=2)
    # print(a)
    ##后获取句法分析中的高频动词名词
    import jpype
    Term = jpype.JClass("com.hankcs.hanlp.seg.common.Term")
    Nature = AttachJVMWrapper("com.hankcs.hanlp.corpus.tag.Nature")
    NeuralNetworkDependencyParser = AttachJVMWrapper(
        "com.hankcs.hanlp.dependency.nnparser.NeuralNetworkDependencyParser")
    stopwords = set()
    with open("../data/dic/stopwords") as f:
        for line in f.readlines():
            stopwords.add(line.strip())
    for i in range(len(pairs_all)):
        pairs = pairs_all[i]
        position = position_all[i]

        jTokens = jpype.java.util.ArrayList()
        for pair in pairs:
            nature = Nature.fromString(pair.flag)
            if nature is None:
                nature = Nature.fromString("n")
            jTokens.add(Term(pair.word, nature))
        tree = NeuralNetworkDependencyParser.compute(jTokens)
        array = tree.wordArray

        def getIdList(entity):
            id_list = []
            id_list.append(entity.ID)
            while entity.HEAD.ID != 0:
                id_list.append(entity.HEAD.ID)
                entity = entity.HEAD
            return id_list

        entity1 = array[position[0]]
        entity2 = array[position[1]]
        id1 = getIdList(entity1)
        id2 = getIdList(entity2)

        remains1, remains2 = len(id1), len(id2)
        tree_word_id = set()
        while remains1 > 0 and remains2 > 0:
            if id1[-remains1] == id2[-remains2]:
                tree_word_id.add(id1[-remains1])
                break
            if (remains1 > remains2):
                tree_word_id.add(id1[-remains1])
                remains1 -= 1
            else:
                tree_word_id.add(id2[-remains2])
                remains2 -= 1
        tree_word_id -= set([entity1.ID, entity2.ID])
        noun_and_verb = map(lambda x: x.LEMMA, filter(lambda x: x.POSTAG.startswith("n") or x.POSTAG.startswith("v"),
                                                      map(lambda id: array[id - 1],
                                                          sorted(tree_word_id, reverse=False))))
        words = list(filter(lambda word: not stopwords.__contains__(word), noun_and_verb))
        result = ''
        for word in words:
            result += word + " "
        relations_detail.append(result.strip())
    return relations_detail


if __name__ == '__main__':
    # pprint(relations_en)
    generateRelationWord()
