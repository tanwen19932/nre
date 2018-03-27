#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : relation.py
# @Author: TW
# @Date  : 2018/3/27
# @Desc  :
from pprint import pprint

from tw_segment.jiebaseg import segOnly

relations_en = ['Component-Whole(e2,e1)', 'Content-Container(e1,e2)', 'Product-Producer(e2,e1)', 'Other',
                'Instrument-Agency(e2,e1)', 'Entity-Destination(e1,e2)', 'Entity-Origin(e1,e2)',
                'Instrument-Agency(e1,e2)',
                'Cause-Effect(e1,e2)', 'Product-Producer(e1,e2)', 'Member-Collection(e1,e2)', 'Message-Topic(e2,e1)',
                'Entity-Origin(e2,e1)', 'Component-Whole(e1,e2)', 'Cause-Effect(e2,e1)', 'Content-Container(e2,e1)',
                'Member-Collection(e2,e1)', 'Entity-Destination(e2,e1)', 'Message-Topic(e1,e2)']
relations_zh = ['部分-整体(e1,e2)', '部分-整体(e2,e1)', '内容-容器(e1,e2)', '内容-容器(e2,e1)', '产品-生产者(e1,e2)', '产品-生产者(e2,e1)',
                '成员-组织(e1,e2)', '成员-组织(e2,e1)', '工具-代理(e1,e2)', '工具-代理(e2,e1)', '起因-影响(e1,e2)', '起因-影响(e2,e1)',
                '消息-话题(e1,e2)', '消息-话题(e2,e1)', '无']
relation_word_dic_zh = {}

def getRelationWord(relation):
    result = []
    try:
        with open("../data/relation/"+relation+".txt","r") as f:
            for line in f.readlines():
                result.append(line)
    except:
        pass
    return result

for relation_zh in relations_zh:
    relation_word_dic_zh[relation_zh]= getRelationWord(relation_zh)

def generateRelationWord():
    #首先进行读取文件 后进行hdp聚类
    sentence_list = []
    corpus = []
    with open("../data/train_zh.txt", "r") as f:
        for line in f.readlines():
            sentence_list.append(line.split("|")[1])
    words_list = list(map(lambda x:map(lambda x:x.word,segOnly(x)),sentence_list))
    from gensim import corpora
    dictionary = corpora.Dictionary(words_list)
    for words in words_list:
        corpus.append(dictionary.doc2bow(words))
    from gensim.models import HdpModel
    hdp = HdpModel(corpus, dictionary)
    hdp.print_topics(num_topics=20, num_words=10)

if __name__ == '__main__':
    pprint(relation_word_dic_zh)
    generateRelationWord()