#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_zh.py
# @Author: TW
# @Date  : 2018/4/27
# @Desc  :
from tw_word2vec.bilstm_trainer_zh import BiLstmTrainer
from tw_word2vec.inputer import Configuration, Inputer
from tw_word2vec.lstm_trainer_zh import LstmTrainer
from tw_word2vec.output_zh import OutPuter

from tw_word2vec.outputer import Outputer
from tw_word2vec.trainer import Trainer

if __name__ == '__main__':
    config = Configuration(
        position_matrix_file_path="../data/posi_matrix.npy",
        word2vec_file_path="../data/needed_zh_word2vec.pkl",
        POS_list_file_path="../data/relation_military/pos_list.txt",
        types_file_path="../data/relation_military/relations_zh.txt",
        corpus_file_path="../data/relation_military/train_zh.txt",
        model_file_path="../data/model/re_military_zh_model.bilstm.hdf5",
    )
    inputer = Inputer(config)
    trainer = Trainer(inputer, BiLstmTrainer())
    outputer = Outputer(trainer)
    predict_texts = ["<loc>美国</loc>目前共有2级11艘航空母舰，包括企业级核动力航母1艘，尼米兹级核动力航母10<loc>艘，</loc>全部采用核动力发动机",
                     "<loc>美国</loc>经过多年航空母舰的发<loc>展，</loc>一直以来都是全球拥有最多、排水量和体积最大、舰载机搭载数量最多、作战效率最强大、而且全部使用核动力航空母舰的国家"]
    import json

    print(json.dumps(outputer.getDescription(predict_texts), ensure_ascii=False))
    corpus = []
    with open("../data/test_zh_marked.txt", "w", encoding="UTF-8") as wf:
        with open("../data/test_zh_by_llc.txt", "r", encoding="UTF-8") as f:
            for line in f.readlines():
                corpus.append(line.strip())
                if (len(corpus) >= 100):
                    output = outputer.getDescription(corpus)
                    corpus.clear()
                    for out in output:
                        wf.write(out["predict_type"]+"|"+out["sentence"]+"|"+out["relation_detail"]+"\n")
        f.close()
    wf.close()





