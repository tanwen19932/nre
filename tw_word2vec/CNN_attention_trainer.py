import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Flatten, Dot, Activation, Permute
from keras.layers import MaxPooling1D, Dropout, regularizers, Conv1D
from keras.layers.core import RepeatVector, Reshape
from keras.models import Model

from tw_segment.en_seg import EnSegmentor
from tw_word2vec.inputer import SentencesVector, Configuration, Inputer
from tw_word2vec.outputer import Outputer
from tw_word2vec.trainer import Trainer
from tw_word2vec.bilstm_attention_trainer import BiLstmAttentionTrainer


class CNN_attention_trainer():

    def buildModel(self, seqLength=100, embeddingDim=300, postionShape2=20, types=19, filters=150, kernel_size=3):
        sequence_input = Input(shape=(seqLength, embeddingDim), dtype='float32',
                                   name="sequence_input")  # 100*1最多100个词组成输入
        # embedded_sequences = inputer.getWordEmbedding()(sequence_input)  # 句子转为向量矩阵 训练集大小*100*300维
        # model test2
        posi_input = Input(shape=(seqLength, postionShape2),
                           name="posi_input")

        typeInput = Input(shape=(types,),
                           name="typeInput")
        # 假设输入embedded_sequeces中一个单词的向量长度为d，假设句长为s，关系种类的向量长度为r
        embedded_sequences = keras.layers.concatenate([sequence_input, posi_input])
        embedded_sequences = Dropout(0.5)(embedded_sequences)
        # 下面的部分就是输入部分应用attention,这里不清楚三个矩阵相乘怎么写，主要是bias的问题
        # 在这里加bias应该是一样的，因为后面乘R的时候，R都是one-hot的，所以bias只不过多训练了几个参数，关键位置应该是一样的
        # 这是probs输出的维度应该是s * r
        probs = Dense(types, use_bias=True)(embedded_sequences)
        # 这里是probs的每一项与当前句子的关系做点积，输出维度是 s * 1
        probsWithRela = Dot(axes=[2, 1])([probs, typeInput])
        # 对输出的probs归一化，此时输出的维度仍然是s * 1
        probs_softmax = Activation(activation="softmax")(probsWithRela)
        input_permute = Permute((2, 1))(embedded_sequences)
        # 将probs拷贝d份，每一份与维度倒置的输入相乘，相当于倒置的输入乘了一个对角矩阵，输出维度是d*s
        probs_repeated = RepeatVector(embeddingDim + postionShape2)(probs_softmax)
        realInput = keras.layers.Multiply()([input_permute, probs_repeated])
        # 将上面的输出倒置过来，维度变成s * d,再传入卷积层
        realInput = Permute((2,1))(realInput)
        input_attention = Dropout(0.5)(realInput)
        # 这时输出应该是(s - kernel_size) * filters，这时候每个kernal中w的维度是 kernal_size * d
        c1 = Conv1D(filters, kernel_size, use_bias=True, activation='tanh')(input_attention)
        # 这时输出的维度应该只是一个一维的filters
        c1 = MaxPooling1D(pool_size=seqLength - kernel_size)(c1)
        c1 = Permute((2, 1))(c1)
        c1 = Reshape((filters,))(c1)
        preds = Dense(types, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                      activity_regularizer=regularizers.l1(0.001))(c1)  # softmax分类
        model = Model(inputs=[sequence_input, posi_input, typeInput], outputs=preds)

        print(model.summary())
        return model


    def train(self, sentences_vector: SentencesVector):
        inputer = sentences_vector.inputer
        config = inputer.config

        model = self.buildModel(config.MAX_SEQUENCE_LENGTH, config.EMBEDDING_DIM, sentences_vector.position_vec.shape[2],
                                    len(inputer.types), filters=150, kernel_size=3)

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
                   'typeInput': sentences_vector.classifications_vec},
                  sentences_vector.classifications_vec,
                  batch_size= 50,
                  epochs=50000,
                  validation_split=0.2,
                  # validation_data=({'sequence_input': x_test, 'posi_input': x_test_posi}, y_test),
                  callbacks=callbacks_list)
        return model


if __name__ == "__main__":
    c = CNN_attention_trainer()
    c.buildModel()