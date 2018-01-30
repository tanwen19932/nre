#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : tf_idf.py
# @Author: TW
# @Date  : 2018/1/30
# @Desc  :
import pandas as pd

import jieba
import jieba.posseg as pseg
import os
import sys

from keras.preprocessing.text import Tokenizer
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def get_tfidf(corpus):
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    return word, weight


if __name__ == "__main__":

    filepath = "~/IdeaProjects/nre/data/train.txt"

    corpus = list()
    type_index = list()
    train = pd.read_csv(filepath_or_buffer=filepath, delimiter='|',
                        # header=["type","e1","e2","doc"],
                        names=["type", "e1", "e2", "doc"]
                        )
    texts = train[['type', 'doc']]
    for type in train['type'].unique():
        list = texts[texts["type"] == type]["doc"].values.tolist()
        from functools import reduce

        type_all_text = reduce(lambda x, y: x +" "+ y, list)
        type_index.append(type)

        from tw_sklearn.my_nltk import  tokenize_and_stem
        corpus.append(reduce(lambda x, y: x +" "+ y, tokenize_and_stem(type_all_text)))

    word, weight = get_tfidf(corpus=corpus)
    total_tfidf_dic = dict()
    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        one_dic= {}
        for j in range(len(word)):
            if weight[i][j] > 0.1:
                one_dic[word[j]]= weight[i][j]
        total_tfidf_dic[type_index[i]]=one_dic

    print(total_tfidf_dic)
    import json
    to_json = json.dumps(total_tfidf_dic)
    print(to_json)