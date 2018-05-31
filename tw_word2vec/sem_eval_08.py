#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : sem_eval_08.py
# @Author: TW
# @Date  : 2018/5/23
# @Desc  : 不可用！！

import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.layers import MaxPooling1D, Dropout, regularizers, Conv1D
from keras.models import Model

from tw_word2vec.bilstm_trainer_zh import BiLstmTrainer
from tw_word2vec.inputer import SentencesVector, Configuration, Inputer
from tw_word2vec.outputer import Outputer
from tw_word2vec.trainer import Trainer


class CnnTrainerEn():
    def train(self, sentences_vector: SentencesVector):
        inputer = sentences_vector.inputer
        config = inputer.config
        sequence_input = Input(shape=(config.MAX_SEQUENCE_LENGTH,), dtype='int32',
                               name="sequence_input")  # 100*1最多100个词组成输入
        embedded_sequences = inputer.getWordEmbedding()(sequence_input)  # 句子转为向量矩阵 训练集大小*100*64维
        # model test2
        posi_input = Input(shape=(config.MAX_SEQUENCE_LENGTH, sentences_vector.position_vec.shape[2]), name="posi_input")
        # pos_input = Input(shape=(config.MAX_SEQUENCE_LENGTH,sentences_vector.pos_vec.shape[2]), name="pos_input")
        embedded_sequences = keras.layers.concatenate([embedded_sequences, posi_input])
        c1 = Conv1D(filters=10, kernel_size=3,
                    activation='relu')(embedded_sequences)
        c1 = MaxPooling1D(pool_size=3)(c1)
        c1 = Dropout(rate=0.7)(c1)
        c1 = Flatten()(c1)
        c1 = Dense(128, activation='relu')(c1)  # 128全连接
        c1 = Dense(64, activation='relu')(c1)  # 64全连接
        preds = Dense(len(inputer.types), activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                      activity_regularizer=regularizers.l1(0.001))(c1)  # softmax分类
        model = Model(inputs=[sequence_input, posi_input], outputs=preds)
        print(model.summary())
        adam = optimizers.Adam(lr=0.001, decay=0.0001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=["categorical_accuracy"])

        # 如果希望短一些时间可以，epochs调小

        # ModelCheckpoint回调函数将在每个epoch后保存模型到filepath，当save_best_only=True保存验证集误差最小的参数

        checkpoint = ModelCheckpoint(config.model_file_path, monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='min')
        # 当监测值不再改善时，该回调函数将中止训练
        early = EarlyStopping(monitor="val_loss", mode="min", patience=50)

        # 开始训练
        callbacks_list = [checkpoint, early]  # early
        # And trained it via:
        model.fit({'sequence_input': sentences_vector.sentence_vec, 'posi_input': sentences_vector.position_vec,
                   'pos_input': sentences_vector.pos_vec},
                  sentences_vector.classifications_vec,
                  batch_size=sentences_vector.sentence_vec.shape[1],
                  epochs=100,
                  # validation_split=0.2,
                  # validation_data=({'sequence_input': x_test, 'posi_input': x_test_posi}, y_test),
                  callbacks=callbacks_list)
        return model

if __name__ == '__main__':
    config = Configuration(
        position_matrix_file_path="../data/posi_matrix.npy",
        word2vec_file_path="../data/needed_word2vec.bin",
        POS_list_file_path="../data/pos_list.txt",
        types_file_path="../data/relations_en.txt",
        corpus_file_path="../data/train.txt",
        model_file_path="../data/model/re_sem_eval_en_model.cnn.hdf5",
    )
    inputer = Inputer(config)
    trainer = Trainer(inputer, CnnTrainerEn())
    outputer = Outputer(trainer)
    predict_texts = ["<loc>美国</loc>目前共有2级11艘航空母舰，包括企业级核动力航母1艘，尼米兹级核动力航母10<loc>艘，</loc>全部采用核动力发动机",
                     "<loc>美国</loc>经过多年航空母舰的发<loc>展，</loc>一直以来都是全球拥有最多、排水量和体积最大、舰载机搭载数量最多、作战效率最强大、而且全部使用核动力航空母舰的国家"]
    import json

    print(json.dumps(outputer.getDescription(predict_texts), ensure_ascii=False))
