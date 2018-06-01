#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : word2vec_zh.py
# @Author: TW
# @Date  : 2018/3/20
# @Desc  :

import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Bidirectional, LSTM
from keras.layers import regularizers
from keras.models import Model, load_model

from tw_word2vec.inputer import SentencesVector
from tw_word2vec.metric import Metrics


class BiLstmAttentionTrainer():
    def train(self,sentences_vector: SentencesVector):
        inputer = sentences_vector.inputer
        config = inputer.config
        sequence_input = Input(shape=(config.MAX_SEQUENCE_LENGTH,), dtype='int32',
                               name="sequence_input")  # 100*1最多100个词组成输入
        embedded_sequences = inputer.getWordEmbedding()(sequence_input)  # 句子转为向量矩阵 训练集大小*100*64维
        # model test2
        posi_input = Input(shape=(config.MAX_SEQUENCE_LENGTH, sentences_vector.position_vec.shape[2]),
                           name="posi_input")
        pos_input = Input(shape=(config.MAX_SEQUENCE_LENGTH, sentences_vector.pos_vec.shape[2]), name="pos_input")
        embedded_sequences = keras.layers.concatenate([embedded_sequences, posi_input, pos_input])
        c1 = Bidirectional(LSTM(100, input_dtype=[100, 182]))(embedded_sequences)
        preds = Dense(len(inputer.types), activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                      activity_regularizer=regularizers.l1(0.001))(c1)  # softmax分类
        model = Model(inputs=[sequence_input, posi_input, pos_input], outputs=preds)
        print(model.summary())
        adam = optimizers.Adam(lr=0.001, decay=0.0001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=["categorical_accuracy"])

        # 如果希望短一些时间可以，epochs调小

        # ModelCheckpoint回调函数将在每个epoch后保存模型到filepath，当save_best_only=True保存验证集误差最小的参数
        checkpoint = ModelCheckpoint(config.model_file_path, monitor='val_loss', verbose=1, mode='min')
        # 当监测值不再改善时，该回调函数将中止训练
        early = EarlyStopping(monitor="val_loss", mode="min", patience=50)
        metrics = Metrics(sentences_vector)
        # 开始训练
        callbacks_list = [checkpoint, early, metrics]  # early
        # And trained it via:
        model.fit({'sequence_input': sentences_vector.sentence_vec, 'posi_input': sentences_vector.position_vec,
                   'pos_input': sentences_vector.pos_vec},
                  sentences_vector.classifications_vec,
                  batch_size=sentences_vector.sentence_vec.shape[1],
                  epochs=100,
                  validation_split=0.2,
                  # validation_data=({'sequence_input': x_test, 'posi_input': x_test_posi}, y_test),
                  callbacks=callbacks_list)
        return model
