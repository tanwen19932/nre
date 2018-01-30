#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : nltk.py
# @Author: TW
# @Date  : 2018/1/30
# @Desc  :
import nltk
import re
from nltk.stem.snowball import SnowballStemmer

# 这里我定义了一个分词器（tokenizer）和词干分析器（stemmer），它们会输出给定文本词干化后的词集合
# 载入 nltk 的英文停用词作为“stopwords”变量
#nltk.download()
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")


def tokenize_and_stem(text):
    # 首先分句，接着分词，而标点也会作为词例存在
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # 过滤所有不含字母的词例（例如：数字、纯标点）
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if token not in stopwords:
                filtered_tokens.append(token)
    stems = [t  for t in filtered_tokens ]
    return stems


def tokenize_only(text):
    # 首先分句，接着分词，而标点也会作为词例存在
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # 过滤所有不含字母的词例（例如：数字、纯标点）
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if token not in stopwords:
                filtered_tokens.append(token)
    return filtered_tokens
