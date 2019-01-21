import numpy as np

import gensim, jieba
import os
import logging
from keras.preprocessing.text import Tokenizer
from gensim.models import word2vec

if __name__ == '__main__':
    corpus_dir = "../data/raw_corpus/doupocangqiong_tiancantudou.txt"
    corpus_cut_dir = "../data/raw_corpus/doupocangqiong_tiancantudou_cut.txt"
    w2v_save_dir = "../data/doupocangqiu_word2vec.txt"

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    if not os.path.exists(corpus_cut_dir):
        # 分词
        with open(corpus_dir, mode="r", encoding="gbk") as f:
            line = f.read()
            line_cut = jieba.cut(line)
            result = " ".join(line_cut)
            with open(corpus_cut_dir, 'w') as f2:
                f2.write(result)
        f.close()
        f2.close()

    # 分词后训练
    if not os.path.exists(w2v_save_dir):
        sentences = word2vec.LineSentence('../data/raw_corpus/doupocangqiong_tiancantudou_cut.txt')
        '''
        　　在gensim中，word2vec 相关的API都在包gensim.models.word2vec中。和算法有关的参数都在类gensim.models.word2vec.Word2Vec中。算法需要注意的参数有：
    
    　　　　1) sentences: 我们要分析的语料，可以是一个列表，或者从文件中遍历读出。后面我们会有从文件读出的例子。
    
    　　　　2) size: 词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度。
    
    　　　　3) window：即词向量上下文最大距离，这个参数在我们的算法原理篇中标记为c，window越大，则和某一词较远的词也会产生上下文关系。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5,10]之间。
    
    　　　　4) sg: 即我们的word2vec两个模型的选择了。如果是0， 则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型。
    
    　　　　5) hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。
    
    　　　　6) negative:即使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间。这个参数在我们的算法原理篇中标记为neg。
    
    　　　　7) cbow_mean: 仅用于CBOW在做投影的时候，为0，则算法中的xw为上下文的词向量之和，为1则为上下文的词向量的平均值。在我们的原理篇中，是按照词向量的平均值来描述的。个人比较喜欢用平均值来表示xw,默认值也是1,不推荐修改默认值。
    
    　　　　8) min_count:需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值。
    
    　　　　9) iter: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。
    
    　　　　10) alpha: 在随机梯度下降法中迭代的初始步长。算法原理篇中标记为η，默认是0.025。
    
    　　　　11) min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。这部分由于不是word2vec算法的核心内容，因此在原理篇我们没有提到。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。
        '''
        model = word2vec.Word2Vec(
            sentences,
            hs=1,
            min_count=3,
            window=5,
            iter=30,
            size=300)

        model.save(w2v_save_dir)

    model = gensim.models.Word2Vec.load(w2v_save_dir)
    print(model.most_similar(positive=['萧战', '萧炎'], negative=['纳兰'], topn=1))
    # print(model.similarity("恐怖", "如斯"))
    tokenizer = Tokenizer(filters="")

    word2idx = {"_PAD": 0}  # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
    embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
    vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embeddings_matrix[i + 1] = vocab_list[i][1]
    wordList = [key[0] for key in vocab_list]
    tokenizer.fit_on_texts(wordList)
    print(tokenizer.word_counts)
    print(tokenizer.texts_to_sequences(['萧战', '萧炎']))
    print(len(model.wv.vocab))
    print('词表长度：', len(model.wv.vocab))
    # print('爱    对应的词向量为：', model['爱'])
    # print('喜欢  对应的词向量为：', model['喜欢'])
    print('弟弟  和  哥哥的距离（余弦距离）', model.wv.similarity('弟弟', '哥哥'))
    print('爱  和  喜欢的距离cos', model.wv.similarity('萧炎', '喜欢'))
    print('爱  和  喜欢的距离（欧式距离）', model.wv.distance('爱', '喜欢'))
    def print_similar(word):
        print('与 {} 最相近的3个词：'.format(word), model.wv.similar_by_word(word, topn=3))
    print_similar("胸大")
    print_similar("兄弟")
    print_similar("古族")
    print_similar("强者")
    print_similar("萧炎")
    print_similar("萧战")
    print_similar("老婆")
    print_similar("能力")
    print('爱，喜欢，恨 中最与众不同的是：', model.wv.doesnt_match(['爱', '喜欢', '恨']))

    # for key in model.wv.vocab:
    #     print(key)

    # print(model.vocabulary)
