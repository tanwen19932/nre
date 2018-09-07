#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : gensim_hdp.py
# @Author: TW
# @Date  : 2018/3/23
# @Desc  :
from gensim import corpora
from gensim.models import HdpModel

import tw_word2vec.word2vec as tw_w2v

jiebaseg = JieBaTokenizer()
default_model: dict = tw_w2v.get_word2vec_dic("../data/needed_zh_word2vec.bin")
keys = list(default_model.keys())
dictionary = corpora.Dictionary(keys)
corpus = []
with open("../data/rawZhData/news_raw_wc2017-12-19.txt", "r") as f:
    for line in f.readlines():
        line = line.strip()
        bow = dictionary.doc2bow(jiebaseg.segOnly(line))
        corpus.append(bow)


hdp = HdpModel(corpus, dictionary)
hdp.print_topics(num_topics=20, num_words=10)
