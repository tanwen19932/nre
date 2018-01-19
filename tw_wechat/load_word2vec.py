import os

import gensim
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping

print('Indexing word vector')
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


MAX_SEQUENCE_LENGTH = 100 # 每篇句子选取1000个词
MAX_NB_WORDS = 20000 # 将字典设置为含有1万个词
EMBEDDING_DIM = 300 # 词向量维度，300维
VALIDATION_SPLIT = 0.2 # 测试集大小，全部数据的20%

word_vec = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), '../data/word2vec/GoogleNews-vectors-negative300.bin'), binary=True)
print(word_vec.most_similar('dog'))
# print out:
# Indexing word vectors.
# Found 10000 word vectors.
# 目的是得到一份字典(embeddings_index)含有1万个词，每个词对应属于自己的300维向量
embeddings_index = {}
for word, vocab_obj in word_vec.vocab.items():
    if int(vocab_obj.index) < MAX_NB_WORDS:
        embeddings_index[word] = word_vec[word]
del word_vec # 删掉gensim模型释放内存
print('Found %s word vectors.' % len(embeddings_index))
import pickle
f = open('needed_word2vec.bin', 'wb')
f.write(pickle.dump(embeddings_index))
f.close()


print('Processing text dataset')

texts = []  # list of text samples
labels = []  # list of label ids
labels_index = {}  # label与name的对应关系

# 读取数据
path = '../content.csv'
contents = pd.read_csv(path)
contents = contents.dropna()

# 提取文本内容与label
texts = contents['content'].values.tolist()
labels = contents['channel_id'].map(int)
labels = labels.values.tolist()

# 获得label与name的对应关系
tem_labels_index = contents.groupby(['name', 'channel_id']).size().reset_index()
tem_labels_index = tem_labels_index[['channel_id', 'name']].values.tolist()
for idx, name in tem_labels_index:
    labels_index[name] = idx
del contents, tem_labels_index

print('Found %s texts.' % len(texts))

# print out
# Processing text dataset
# Found 57867 texts.

tokenizer = Tokenizer(num_words=MAX_NB_WORDS) # 传入我们词向量的字典
tokenizer.fit_on_texts(texts) # 传入我们的训练数据，得到训练数据中出现的词的字典
sequences = tokenizer.texts_to_sequences(texts) # 根据训练数据中出现的词的字典，将训练数据转换为sequences

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) # 限制每篇文章的长度

labels = to_categorical(np.asarray(labels)) # label one hot表示
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)



# 打乱文章顺序
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

# 切割数据
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


num_words = min(MAX_NB_WORDS, len(word_index))  # 对比词向量字典中包含词的个数与文本数据所有词的个数，取小
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # 文本数据中的词在词向量字典中没有，向量为取0；如果有则取词向量中该词的向量
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(num_words,#词个数
                            EMBEDDING_DIM,#300维向量
                            weights=[embedding_matrix],#向量矩阵
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')# 100*1最多100个词组成输入
embedded_sequences = embedding_layer(sequence_input)# 句子转为向量矩阵100*300维
x = Conv1D(filters=128, kernel_size=5, activation='relu')(embedded_sequences)#成为96*296*128
x = MaxPooling1D(pool_size=5)(x)#成为92*292*128
x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)#成为88*288*128
x = MaxPooling1D(pool_size=5)(x)#84*284*128
x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)#80*280*128
x = MaxPooling1D(pool_size=35)(x)#46*246*128
x = Flatten()(x)
x = Dense(128, activation='relu')(x)#128全连接
preds = Dense(len(labels_index), activation='softmax')(x)#softmax分类

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# 如果希望短一些时间可以，epochs调小

# ModelCheckpoint回调函数将在每个epoch后保存模型到filepath，当save_best_only=True保存验证集误差最小的参数
file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# 当监测值不再改善时，该回调函数将中止训练
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

# 开始训练
callbacks_list = [checkpoint, early] #early

model.fit(x_train, y_train,
          batch_size=128,
          epochs=50,
          validation_data=(x_val, y_val),
          callbacks=callbacks_list)

model.save(filepath="model1.model")


