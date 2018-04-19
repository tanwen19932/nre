#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : jiebaseg.py
# @Author: TW
# @Date  : 2018/3/19
# @Desc  :

import jieba

import jieba.posseg as pseg
from bs4 import Tag, NavigableString, BeautifulSoup


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

def segWithNerTag(sentence:str):
    soup = BeautifulSoup(sentence, "html5lib")
    pairs = []
    e_count = 0
    temp_str = ""
    for tag in soup.body.contents:
        if isinstance(tag, Tag):
            pairs.extend(segOnly(temp_str))
            from jieba.posseg import pair
            pairs.append(pair(tag.text, tag.name))
            if (e_count == 0):
                position_e1 = len(pairs) - 1
            elif (e_count == 1):
                position_e2 = len(pairs) - 1
            temp_str = ""
            e_count += 1
        elif isinstance(tag, NavigableString):
            temp_str += tag
        if e_count > 2:
            break
    if (e_count > 2):
        return None
    if (e_count != 2):
        return None
    if (len(temp_str) > 0):
        pairs.extend(segOnly(temp_str))
    return pairs,(position_e1, position_e2)

def segListWithNerTag(sentences:list):
    pairs_all = []
    position_all = []
    for sentence in sentences:
        try:
            pairs,position_pair = segWithNerTag(sentence)
            pairs_all.append(pairs)
            position_all.append(position_pair)
        except:
            print(sentence)
            pass
    return pairs_all,position_all



if __name__ == '__main__':
    with open("../data/rawZhData/news_raw_wc2017-12-19.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            pairList = segOnly(line)
            print(pairList)
            print()
            # print(list(map(lambda x:x.word,pairList)))
