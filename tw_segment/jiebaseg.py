#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : jiebaseg.py
# @Author: TW
# @Date  : 2018/3/19
# @Desc  :

import jieba

import jieba.posseg as pseg


def segOnly(str):
    totalWord = []
    words = pseg.cut(str)
    for word in words:
        totalWord.append(word)
    return totalWord


def segSpaceSplit(str):
    totalWord = ""
    words = pseg.cut(str)
    for word in words:
        totalWord += word.word + ' '
    return totalWord


def segWithNER(str):
    totalWord = []
    words = pseg.cut(str)
    for word in words:
        totalWord.append(word)
    return totalWord


if __name__ == '__main__':
    with open("../data/rawZhData/news_raw_wc2017-12-19.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            pairList = segWithNER(line)
            print(pairList)
            print()
            # print(list(map(lambda x:x.word,pairList)))
