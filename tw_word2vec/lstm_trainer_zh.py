#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : word2vec_zh.py
# @Author: TW
# @Date  : 2018/3/20
# @Desc  :

from tw_segment import jieba_seg
from tw_word2vec.cnn_input_zh import *

import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import MaxPooling1D, Dropout, regularizers, LSTM
from keras.layers import Dense, Input, Flatten
from keras.models import Model, load_model
import numpy as np

from tw_keras.multi_layer import MultiConv1D
model_path =  "../data/model/re_zh_model.lstm.hdf5"

def train(sentences_vector: SentencesVector):
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name="sequence_input")  # 100*1最多100个词组成输入
    embedded_sequences = embedding_layer(sequence_input)  # 句子转为向量矩阵 训练集大小*100*64维
    # model test2
    posi_input = Input(shape=(MAX_SEQUENCE_LENGTH, 40), name="posi_input")
    pos_input = Input(shape=(MAX_SEQUENCE_LENGTH, len(all_pos_list)), name="pos_input")
    embedded_sequences = keras.layers.concatenate([embedded_sequences, posi_input, pos_input])
    # conv1d_1s = MultiConv1D(filters=[90, 80, 70, 50, 30, 10], kernel_size=[3, 4, 5], activation='relu')
    c1 = LSTM(100, input_dtype=[100, 182])(embedded_sequences)
    # c1 = MaxPooling1D(pool_size=3)(c1)
    # c1 = Dropout(rate=0.7)(c1)
    # c1 = Flatten()(c1)
    # c1 = Dense(128, activation='relu')(c1)  # 128全连接
    # c1 = Dense(64, activation='relu')(c1)  # 64全连接
    preds = Dense(len(types), activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                  activity_regularizer=regularizers.l1(0.001))(c1)  # softmax分类
    model = Model(inputs=[sequence_input, posi_input, pos_input], outputs=preds)
    print(model.summary())
    adam = optimizers.Adam(lr=0.001, decay=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=["accuracy"])

    # 如果希望短一些时间可以，epochs调小

    # ModelCheckpoint回调函数将在每个epoch后保存模型到filepath，当save_best_only=True保存验证集误差最小的参数
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # 当监测值不再改善时，该回调函数将中止训练
    early = EarlyStopping(monitor="val_loss", mode="min", patience=50)

    # 开始训练
    callbacks_list = [checkpoint, early]  # early
    # And trained it via:
    model.fit({'sequence_input': sentences_vector.sentence_vec, 'posi_input': sentences_vector.position_vec,
               'pos_input': sentences_vector.pos_vec},
              sentences_vector.classifications_vec,
              batch_size=128,
              epochs=500,
              # validation_split=0.2,
              # validation_data=({'sequence_input': x_test, 'posi_input': x_test_posi}, y_test),
              callbacks=callbacks_list)
    return model


import os
if not os.path.exists(model_path):
    vector = getSentenceVectorFromFile("../data/train_zh.txt")
    print(vector.sentence_vec)
    print(vector.position_vec)
    print(vector.pos_vec)
    print(vector.classifications_vec)
    model = train(vector)
    model.save(model_path)
model = load_model(model_path)


def predict(sentence_vector: SentencesVector):
    id = model.predict({'sequence_input': sentence_vector.sentence_vec, 'posi_input': sentence_vector.position_vec,
                        'pos_input': sentence_vector.pos_vec})
    output = []
    for row in id:
        max_index = row.argsort()[-1]
        # raw_type = types[y_test[i].argsort()[-1]]
        for i in range(len(row)):
            print(types[i], row[i])
        predict_type = types[max_index]
        output.append(predict_type)
    return output
