import os

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input, Flatten
from keras.models import Model, load_model

from tw_word2vec.keras_input import embedding_layer, MAX_SEQUENCE_LENGTH, types, get_xy, get_sentence_vec


def train():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')# 100*1最多100个词组成输入
    embedded_sequences = embedding_layer(sequence_input)# 句子转为向量矩阵 训练集大小*100*300维
    x = Conv1D(filters=128, kernel_size=5, activation='sigmoid')(embedded_sequences)#卷积层5*300成为 96*128
    x = MaxPooling1D(pool_size=2)(x)#池化层2*128 stride =2 成为 48*128
    x = Conv1D(filters=128, kernel_size=5, activation='sigmoid')(x)#成为44*128
    x = MaxPooling1D(pool_size=3)(x)#46*246*128
    x = Flatten()(x)
    x = Dense(128, activation='sigmoid')(x)#128全连接
    preds = Dense(len(types), activation='softmax')(x)#softmax分类
    model = Model(sequence_input, preds)
    print(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    # 如果希望短一些时间可以，epochs调小

    # ModelCheckpoint回调函数将在每个epoch后保存模型到filepath，当save_best_only=True保存验证集误差最小的参数
    file_path="../data/model/weights_base.best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # 当监测值不再改善时，该回调函数将中止训练
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

    # 开始训练
    callbacks_list = [checkpoint, early] #early
    x_train ,y_train = get_xy("../data/train.txt",0.8)
    x_test ,y_test = get_xy("../data/test.txt")
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=50,
              validation_data=(x_test, y_test),
              callbacks=callbacks_list)
    return model
# model.save(filepath="model1.model")

if __name__ == '__main__':
    # model = train()
    model = load_model("../data/model/weights_base.best.hdf5")
    id = model.predict(get_sentence_vec("She was looking beautiful in the dim light that emanated from my broken table-lamp"))
    print(id)
    # print(types[int(id)])