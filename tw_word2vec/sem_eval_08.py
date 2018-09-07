#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : sem_eval_08.py
# @Author: TW
# @Date  : 2018/5/23
# @Desc  :

import keras
import os
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.layers import MaxPooling1D, Dropout, regularizers, Conv1D
from keras.models import Model

from tw_segment.en_seg import EnSegmentor
from tw_word2vec.inputer import SentencesVector, Configuration, Inputer
from tw_word2vec.outputer import Outputer
from tw_word2vec.trainer import Trainer
from tw_word2vec.bilstm_attention_trainer import BiLstmAttentionTrainer
# from ltl_pytorch import ACNN_trainer



class CnnTrainerEn():
    def train(self, sentences_vector: SentencesVector):
        inputer = sentences_vector.inputer
        config = inputer.configcn
        embedded_sequences = Input(shape=(config.MAX_SEQUENCE_LENGTH, config.EMBEDDING_DIM*3), dtype='float32',
                               name="sequence_input")  # 100*1最多100个词组成输入
        # embedded_sequences = inputer.getWordEmbedding()(sequence_input)  # 句子转为向量矩阵 训练集大小*100*300维
        # model test2
        posi_input = Input(shape=(config.MAX_SEQUENCE_LENGTH, sentences_vector.position_vec.shape[2]),
                           name="posi_input")
        # pos_input = Input(shape=(config.MAX_SEQUENCE_LENGTH,sentences_vector.pos_vec.shape[2]), name="pos_input")
        embedded_sequences = keras.layers.concatenate([embedded_sequences, posi_input])
        c1 = Dropout(rate=0.25)(embedded_sequences)
        c1 = Conv1D(filters=150, kernel_size=3,
                    activation='relu')(c1)
        c1 = MaxPooling1D(pool_size=98)(c1)
        c1 = Dropout(rate=0.25)(c1)
        c1 = Flatten()(c1)
        c1 = Dense(128, activation='relu')(c1)  # 128全连接
        preds = Dense(len(inputer.types), activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                      activity_regularizer=regularizers.l1(0.001))(c1)  # softmax分类
        model = Model(inputs=[embedded_sequences, posi_input], outputs=preds)
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
        early = EarlyStopping(monitor="val_loss", mode="min", patience=500)

        # 开始训练
        callbacks_list = [checkpoint, early]  # early
        # And trained it via:
        model.fit({'sequence_input': sentences_vector.embedded_sequences, 'posi_input': sentences_vector.position_vec,
                   'pos_input': sentences_vector.pos_vec},
                  sentences_vector.classifications_vec,
                  batch_size= 50,
                  epochs=50000,
                  validation_split=0.2,
                  # validation_data=({'sequence_input': x_test, 'posi_input': x_test_posi}, y_test),
                  callbacks=callbacks_list)
        return model


if __name__ == '__main__':
    testType = input("CNN?RNN:")
    config = Configuration(
        word_segmentor=EnSegmentor(),
        EMBEDDING_DIM=300,
        position_matrix_file_path="../data/posi_matrix.npy",
        word2vec_file_path="../data/needed_word2vec.pkl",
        POS_list_file_path="../data/pos_list.txt",
        types_file_path="../data/relations_en.txt",
        corpus_file_path="../data/train_en.txt",
        model_file_path="../data/model/re_sem_eval_en_model.cnn.hdf5",
    )
    inputer = Inputer(config)
    trainer = Trainer(inputer, testType)
    outputer = Outputer(trainer)
    # outputer.getEvaluation(testType)
    predict_texts = [" <e1>level</e1> of experience has already been mentioned in the previous <e2>chapter</e2>.",
                     " <e1>level</e1> of experience has already been mentioned in the previous <e2>chapter</e2>."]
    import json
    # print(json.dumps(outputer.getDescription(predict_texts, testType), ensure_ascii=False))
