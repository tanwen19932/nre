#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : word2vec_zh.py
# @Author: TW
# @Date  : 2018/3/20
# @Desc  :

import os

from keras.models import load_model

from tw_word2vec.inputer import Inputer, Configuration, SentencesVector
from tw_word2vec.lstm_trainer_zh import LstmTrainer
from tw_word2vec.outputer import Outputer


class Trainer(object):
    def __init__(self, inputer: Inputer, modelTrainer: object) -> object:
        self.config = inputer.config
        self.inputer = inputer
        self.modelTrainer = modelTrainer
        config = self.config
        if not os.path.exists(config.model_file_path):
            vector = inputer.getSentenceVectorFromFile(config.corpus_file_path)
            print(vector.sentence_vec.shape)
            print(vector.position_vec.shape)
            print(vector.pos_vec.shape)
            print(vector.classifications_vec.shape)
            self.model = self.train(vector)
            self.model.save(config.model_file_path)
        self.model = load_model(config.model_file_path)

    def train(self,sentence_vector: SentencesVector):
        return self.modelTrainer.train(sentence_vector)

    def predict(self, sentence_vector: SentencesVector):
        id = self.model.predict(
            {'sequence_input': sentence_vector.sentence_vec, 'posi_input': sentence_vector.position_vec,
             'pos_input': sentence_vector.pos_vec})
        output = []
        for row in id:
            max_index = row.argsort()[-1]
            # raw_type = types[y_test[i].argsort()[-1]]
            for i in range(len(row)):
                print(inputer.types[i], row[i])
            predict_type = inputer.types[max_index]
            output.append(predict_type)
        return output


if __name__ == '__main__':
    config = Configuration(
        position_matrix_file_path="../data/posi_matrix.npy",
        word2vec_file_path="../data/needed_zh_word2vec.bin",
        POS_list_file_path="../data/military/pos_list.txt",
        types_file_path="../data/military/relations_zh.txt",
        corpus_file_path="../data/military/train_zh.txt",
        model_file_path="../data/model/re_military_zh_model.lstm.hdf5",
    )
    inputer = Inputer(config)
    trainer = Trainer(inputer,LstmTrainer())
    outputer = Outputer(trainer)
    predict_texts = ["所以美军派出<aircraft>“里根”号</aircraft>航母来<loc>南海</loc>增援。",
                     "4月23日,<army>日本海上自卫队</army>“足柄”号和“五月雨”号两艘驱逐舰与<aircraft>“卡尔·文森”号</aircraft>航母打击群在西太平洋举行联合训练。"]
    import json
    print(json.dumps(outputer.getDescription(predict_texts), ensure_ascii=False))