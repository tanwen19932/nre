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

from keras import backend as K

from tw_word2vec.inputer import SentencesVector
from tw_word2vec.metric import Metrics

from tw_word2vec.attention_utils import get_activations, get_data_recurrent
INPUT_DIM = 300
TIME_STEPS = 100
lstm_units = 100
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False

class BiLstmAttentionTrainer():
    # 只是测试用
    def model_attention_applied_after_lstm(self):
        inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
        d = Dropout(0.3)(inputs)
        lstm_units = 100
        lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(d)
        attention_mul = self.oldAttention(lstm_out)
        output = Dense(21, activation='softmax')(attention_mul)
        model = Model(inputs=[inputs], outputs=output)
        return model
    # 只是测试用
    def model_attention_applied_before_lstm(self):
        inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
        attention_mul = self.oldAttention(inputs)
        lstm_units = 100
        attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
        output = Dense(21, activation='softmax')(attention_mul)
        model = Model(input=[inputs], output=output)
        return model

    # 这个新实现的Attention的参照论文是A Multi-attention-Based Bidirectional Long Short-Term Memory Network for Relation Extraction
    def newAttention(self, inputs):
        # tanh激活，该层输出维度t * d
        inputs_tanh = Activation('tanh', name="tanh")(inputs)
        # 该层dense的w参数应该为1 * d，没有加偏置项，最后输出维度 1 * t,这里的全连接层实际输出为t * 1
        probs = Dense(1, activation="softmax", name="softmax")(inputs_tanh)
        # 该层reshape一下，保证维度是t，而不是t * 1
        probs_reshape = Reshape((self.config.MAX_SEQUENCE_LENGTH,))(probs)
        # 输入inputs 为 t * d，t是时间步数，d为lstm输出的维度，在permute层将其转置为d * t
        inputs_transpose = Permute((2, 1), name="input_tranpose")(inputs)
        # 这里需要让 d * t(转置后的输入)，与probs_repeat（d * t）做内积，变成d * 1
        output_attention_mul = keras.layers.Dot(axes=[2, 1])([inputs_transpose, probs_reshape])
        # output_attention_mul = keras.layers.Multiply()([inputs_transpose, probs_repeat])
        return output_attention_mul

    # 这个旧的Attention实现相对简单，训练一个全连接层表示每个单词的权重
    def oldAttention(self, inputs):
        # inputs.shape = (batch_size, time_steps, input_dim)
        input_dim = int(inputs.shape[2])
        # 这个重排的意义在于试着通过dense表达出哪个time_steps比较重要，即这次输入attention应该放在哪个单词上
        a = Permute((2, 1))(inputs)  # 维度重排，应该是input_dim跟time_steps换一下，batch_size不变
        a = Reshape((input_dim, TIME_STEPS))(a)  # this line is not useful. It's just to know which dimension is what.
        # 这一层就是把前一个RNN输出的向量经过一个matrix，再加个softmax，产生的结果可以作为下一层输入
        a = Dense(TIME_STEPS, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(a)
        if SINGLE_ATTENTION_VECTOR:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        # 这是返回inputs和a_probs逐元素积的张量
        output_attention_mul = keras.layers.Multiply()([inputs, a_probs])
        # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')

        output_attention = Flatten()(output_attention_mul)
        return output_attention

    def train(self, sentences_vector: SentencesVector):
        print("构建模型")
        self.inputer = sentences_vector.inputer
        self.config = self.inputer.config
        # 获得输入的嵌入向量
        embedded_sequences = Input(shape=(self.config.MAX_SEQUENCE_LENGTH, self.config.EMBEDDING_DIM*3), dtype='float32',
                               name="sequence_input")  # 100*1最多100个词组成输入
        # embedded_sequences = inputer.getWordEmbedding()(sequence_input)  # 句子转为向量矩阵 训练集大小*100*64维
        posi_input = Input(shape=(self.config.MAX_SEQUENCE_LENGTH, sentences_vector.position_vec.shape[2]),
                           name="posi_input")
        pos_input = Input(shape=(self.config.MAX_SEQUENCE_LENGTH, sentences_vector.pos_vec.shape[2]), name="pos_input")
        input_embed = keras.layers.concatenate([embedded_sequences, posi_input, pos_input])
        d = Dropout(0.5)(input_embed)
        # LSTM获取每个timesteps的特征
        lstm_units = 100
        lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)), merge_mode='concat')(d)
        lstm_out_d = Dropout(0.5)(lstm_out)
        # 旧的attention机制
        # attention_mul = oldAttention(lstm_out_d)
        # attention_mul_d = Dropout(0.2)(attention_mul)

        # 新的attention机制
        attention_mul = self.newAttention(lstm_out_d)
        f = Dropout(0.5)(attention_mul)
        # 输出层，softmax
        output = Dense(len(self.inputer.types), activation='softmax', kernel_regularizer=regularizers.l2(0.01))(f)
        # 模型
        model = Model(inputs=[embedded_sequences , posi_input, pos_input], outputs=output)
        print(model.summary())
        # 优化器
        adam = optimizers.Adam(lr=0.005)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=["categorical_accuracy"])
        # 训练
        # ModelCheckpoint回调函数将在每个epoch后保存模型到filepath，当save_best_only=True保存验证集误差最小的参数
        checkpoint = ModelCheckpoint(self.config.model_file_path, monitor='val_acc', verbose=1, mode='max', save_best_only= True)
        # 当监测值不再改善时，该回调函数将中止训练
        early = EarlyStopping(monitor="val_acc", mode="min", patience=500)
        metrics = Metrics(sentences_vector)
        # 开始训练
        callbacks_list = [checkpoint, early]  # early
        # And trained it via:
        hist = model.fit({'sequence_input': sentences_vector.embedded_sequences, 'posi_input': sentences_vector.position_vec,
                   'pos_input': sentences_vector.pos_vec},
                  sentences_vector.classifications_vec,
                  batch_size=100,
                  epochs=3000,
                  validation_split=0.2,
                  # validation_data=({'sequence_input': x_test, 'posi_input': x_test_posi}, y_test),
                  callbacks=callbacks_list)
        with open('history.txt', 'w') as f:
            f.write(str(hist.history))
        return model

if __name__ == '__main__':
    N = 300000
    b = BiLstmAttentionTrainer()
    m = b.model_attention_applied_after_lstm()
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(m.summary())