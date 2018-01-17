import numpy as np
import os
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), '../data/word2vec/GoogleNews-vectors-negative300.bin'), binary=True)
print(model.most_similar('dog'))
