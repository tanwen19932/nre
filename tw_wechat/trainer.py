import os

import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, regularizers
from keras.layers import Dense, Input, Flatten
from keras.models import Model, load_model
import numpy as np
import numpy as np
import heapq

from tw_keras import kerasf1
from tw_keras.multi_layer import MultiConv1D
from tw_word2vec.keras_input import embedding_layer, MAX_SEQUENCE_LENGTH, types, get_xy, get_sentence_vec


def train():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name="sequence_input")  # 100*1最多100个词组成输入
    embedded_sequences = embedding_layer(sequence_input)  # 句子转为向量矩阵 训练集大小*100*300维
    # model test2
    posi_input = Input(shape=(MAX_SEQUENCE_LENGTH, 40), name="posi_input")
    embedded_sequences = keras.layers.concatenate([embedded_sequences, posi_input])
    conv1d_1s = MultiConv1D(filters=[90, 80, 70, 50, 30, 10], kernel_size=[3, 4, 5], activation='sigmoid')
    best_model = None
    count = 0
    for conv1d in conv1d_1s:
        c1 = conv1d(embedded_sequences)
        c1 = MaxPooling1D(pool_size=3)(c1)
        c1 = Dropout(rate=0.6)(c1)
        c1 = Flatten()(c1)
        # c1 = Dense(128, activation='relu')(c1)  # 128全连接
        # c1 = Dense(64, activation='relu')(c1)  # 64全连接
        preds = Dense(len(types), activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                      activity_regularizer=regularizers.l1(0.001))(c1)  # softmax分类
        model = Model(inputs=[sequence_input, posi_input], outputs=preds)
        print(model.summary())
        adam = optimizers.Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=["categorical_accuracy"])

        # 如果希望短一些时间可以，epochs调小

        # ModelCheckpoint回调函数将在每个epoch后保存模型到filepath，当save_best_only=True保存验证集误差最小的参数
        file_path = "../data/model/weights_base.temp" + str(count) + ".hdf5"
        checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        # 当监测值不再改善时，该回调函数将中止训练
        early = EarlyStopping(monitor="val_loss", mode="min", patience=50)

        # 开始训练
        callbacks_list = [checkpoint, early]  # early
        x_train, x_train_posi, y_train, x_test, x_test_posi, y_test = get_xy("../data/train.txt", 0.8)

        # And trained it via:
        model.fit({'sequence_input': sequence_input, 'posi_input': posi_input},
                  y_train,
                  batch_size=128,
                  epochs=200,
                  validation_data=(x_test, y_test),
                  callbacks=callbacks_list)
        print(model)
        count += 1
        best_model = model
    return best_model


# model.save(filepath="model1.model")

if __name__ == '__main__':
    model = train()
    filepath = "../data/model/weights_base.best.hdf5"
    model.save(filepath)
    model = load_model(filepath)
    doc_vec = get_sentence_vec(
        ["The most common audits were about waste and recycling"
            , "The company fabricates plastic chairs"
            , "The school master teaches the lesson with a stick "
         ])
    print(doc_vec.shape)
    x_test, x_posi, y_test = get_xy("../data/test.txt")
    id = model.predict(x_test)
    print("x_test 2:", x_test[0])
    i = 0
    right = 0
    for row in id:
        if i == 0:
            print(row)
        max_index = row.argsort()[-1]
        raw_type = types[y_test[i].argsort()[-1]]
        predict_type = types[max_index]
        is_right = "错误"
        if raw_type.__eq__(predict_type):
            is_right = "正确"
            right += 1
        else:
            print(raw_type, max_index, predict_type, is_right, float(right / i))
        i += 1
