#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_zh.py
# @Author: TW
# @Date  : 2018/4/27
# @Desc  :
from tw_word2vec.lstm_trainer_zh import LstmTrainer
from tw_word2vec.output_zh import OutPuter

if __name__ == '__main__':
    corpus = []
    outputer = OutPuter(LstmTrainer())
    with open("../data/test_zh_marked.txt", "w") as wf:
        with open("../data/test_zh_by_llc.txt", "r") as f:
            for line in f.readlines():
                corpus.append(line.strip())
                if (len(corpus) >= 100):
                    output = outputer.getDescription(corpus)
                    corpus.clear()
                    for out in output:
                        wf.write(out["predict_type"]+"|"+out["sentence"]+"|"+out["relation_detail"]+"\n")
        f.close()
    wf.close()




