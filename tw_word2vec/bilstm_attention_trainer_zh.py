#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : word2vec_zh.py
# @Author: TW
# @Date  : 2018/3/20
# @Desc  :

import keras
from keras import optimizers
from keras.engine import InputSpec
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Bidirectional, LSTM, merge
from keras.layers.core import *
from keras.models import *

from tw_word2vec.inputer import SentencesVector
from tw_word2vec.metric import Metrics

from tw_word2vec.attention_utils import get_activations, get_data_recurrent
INPUT_DIM = 128
TIME_STEPS = 100
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(100,128,))
    lstm_units = 100
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(21, activation='softmax')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(21, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


class BiLstmAttentionTrainer():
    def train(self, sentences_vector: SentencesVector):
        print("构建模型")
        inputer = sentences_vector.inputer
        config = inputer.config
        # 获得input_embed
        sequence_input = Input(shape=(config.MAX_SEQUENCE_LENGTH,), dtype='int32',
                               name="sequence_input")  # 100*1最多100个词组成输入
        embedded_sequences = inputer.getWordEmbedding()(sequence_input)  # 句子转为向量矩阵 训练集大小*100*64维
        posi_input = Input(shape=(config.MAX_SEQUENCE_LENGTH, sentences_vector.position_vec.shape[2]),
                           name="posi_input")
        pos_input = Input(shape=(config.MAX_SEQUENCE_LENGTH, sentences_vector.pos_vec.shape[2]), name="pos_input")
        input_embed = keras.layers.concatenate([embedded_sequences, posi_input, pos_input])

        lstm_units = 100
        lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat' )(input_embed)
        attention_mul = attention_3d_block(lstm_out)
        attention_mul = Flatten()(attention_mul)
        output = Dense(len(inputer.types), activation='softmax')(attention_mul)
        model = Model(inputs=[sequence_input, posi_input, pos_input], outputs=output)
        print(model.summary())

        adam = optimizers.Adam(lr=0.001, decay=0.0001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=["categorical_accuracy"])

        # ModelCheckpoint回调函数将在每个epoch后保存模型到filepath，当save_best_only=True保存验证集误差最小的参数
        checkpoint = ModelCheckpoint(config.model_file_path, monitor='val_loss', verbose=1, mode='min', save_best_only= True)
        # 当监测值不再改善时，该回调函数将中止训练
        early = EarlyStopping(monitor="val_loss", mode="min", patience=50)
        metrics = Metrics(sentences_vector)
        # 开始训练
        callbacks_list = [checkpoint, early]  # early
        # And trained it via:
        model.fit({'sequence_input': sentences_vector.sentence_vec, 'posi_input': sentences_vector.position_vec,
                   'pos_input': sentences_vector.pos_vec},
                  sentences_vector.classifications_vec,
                  batch_size=100,
                  epochs=100,
                  validation_split=0.2,
                  # validation_data=({'sequence_input': x_test, 'posi_input': x_test_posi}, y_test),
                  callbacks=callbacks_list)
        return model


if __name__ == '__main__':

    N = 300000
    m = model_attention_applied_after_lstm()
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(m.summary())