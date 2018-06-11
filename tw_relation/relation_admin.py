#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : relation.py
# @Author: TW
# @Date  : 2018/3/27
# @Desc  :
from pprint import pprint

from tw_segment.jieba_seg import *

ZH_RELATION_PATH = "../data/relations_zh"
EN_RELATION_PATH = "../data/relations_en"

segmentor = JieBaTokenizer()

def getFileLines(file_path):
    result = []
    try:
        with open(file_path, "r", encoding="UTF-8") as f:
            for line in f.readlines():
                line = line.strip()
                if (len(line) > 0):
                    result.append(line)
    except:
        pass
    return result


def getRelationWord(relation):
    file_path = "../data/relation/" + relation + ".txt"
    return getFileLines(file_path)


class RelationWordAdmin(object):
    def __init__(self,relation_path) -> None:
        self.relations = getFileLines(relation_path)
        self.relation_word_dic = {}
        for relation in self.relations:
            self.relation_word_dic[relation] = getRelationWord(relation)
        # print(self.relation_word_dic)

    def getRelationDetail(self, paris_all, position_all,predict_types):
        detail = []
        for i in range(len(position_all)):
            pos = position_all[i]
            predict_type = predict_types[i]
            if predict_type not in self.relations:
                detail.append("")
                continue
            is_add = False
            # for pair in paris_all[i][pos[0]+1:pos[1]]:
            #     print(pair)
            #     if pair.word in self.relation_word_dic[predict_type]:
            #         detail.append(pair.word)
            #         is_add=True
            #         break
            for pair in paris_all[i][pos[0]+1:pos[1]]:
                print(pair)
                print(self.relation_word_dic)
                if pair[0] in self.relation_word_dic[predict_type]:
                    detail.append(pair[0])
                    is_add=True
                    break
            if not is_add:
                detail.append("")
        return detail

relation_admin_zh = RelationWordAdmin(ZH_RELATION_PATH)
relation_admin_en = RelationWordAdmin(EN_RELATION_PATH)




def generateRelationWord(sentence_list: list) -> list:
    # 进行hdp聚类 获取相关词
    # 句法分析获取相关词
    # 统计两者都出现的词 作为待选关系词典
    # hdp
    result_word = []
    hdp_word = getRelationDetailByHDP(sentence_list)
    parse_word = []
    for word in getRelationDetailByParse(sentence_list):
        for temp_word in word.split(" "):
            parse_word.append(temp_word)
    # print(hdp_word)
    # print(parse_word)
    for word_value in hdp_word:
        if(parse_word.__contains__(word_value[0])):
            result_word.append(word_value[0])
    return result_word



from pyhanlp import *
def getRelationDetailByHDP(sentence_list):
    # 聚类获取结果
    corpus = []
    pairs_all, position_all = segmentor.segListWithNerTag(sentence_list)
    words_list = []
    for pairs in pairs_all:
        word_list = []
        for pair in pairs:
            if pair.flag.__contains__("v") or pair.flag.__contains__("n"):
                word_list.append(pair.word)
        words_list.append(word_list)
    # words_list = list(map(lambda pairs: map(lambda x: x.word, pairs), pairs_all))
    from gensim import corpora
    dictionary = corpora.Dictionary(words_list)
    for words in words_list:
        corpus.append(dictionary.doc2bow(words))
    from gensim.models import HdpModel
    hdp = HdpModel(corpus, dictionary)
    a = hdp.print_topics()
    words = {}
    for topic in a:
        word_details = str(topic[1]).split(" + ")
        for  word_detail in word_details:
            word = str(word_detail[word_detail.index("*") + 1:])
            num = float(str(word_detail[:word_detail.index("*")]))
            if not (words.__contains__(word)):
                words[word] = num
            else:
                words[word] += num
    words = sorted(words.items(), key=lambda d: d[1])
    return words  # 后获取句法分析中的高频动词名词)


def getRelationDetailByParse(sentence_list):
    # 从Hanlp获取包含两个实体最小生成树,将树中的所有词加入统计然后与RelationWord中的所有词进行同步取Top 可获取相关内容
    relations_detail = []
    pairs_all, position_all = segmentor.segListWithNerTag(sentence_list)
    import jpype
    Term = jpype.JClass("com.hankcs.hanlp.seg.common.Term")
    Nature = AttachJVMWrapper("com.hankcs.hanlp.corpus.tag.Nature")
    NeuralNetworkDependencyParser = AttachJVMWrapper(
        "com.hankcs.hanlp.dependency.nnparser.NeuralNetworkDependencyParser")
    stopwords = set()
    with open("../data/dic/stopwords") as f:
        for line in f.readlines():
            stopwords.add(line.strip())
    for i in range(len(pairs_all)):
        pairs = pairs_all[i]
        position = position_all[i]

        jTokens = jpype.java.util.ArrayList()
        for pair in pairs:
            nature = Nature.fromString(pair.flag)
            if nature is None:
                nature = Nature.fromString("n")
            jTokens.add(Term(pair.word, nature))
        tree = NeuralNetworkDependencyParser.compute(jTokens)
        array = tree.wordArray

        def getIdList(entity):
            id_list = []
            id_list.append(entity.ID)
            while entity.HEAD.ID != 0:
                id_list.append(entity.HEAD.ID)
                entity = entity.HEAD
            return id_list

        entity1 = array[position[0]]
        entity2 = array[position[1]]
        id1 = getIdList(entity1)
        id2 = getIdList(entity2)

        remains1, remains2 = len(id1), len(id2)
        tree_word_id = set()
        while remains1 > 0 and remains2 > 0:
            if id1[-remains1] == id2[-remains2]:
                tree_word_id.add(id1[-remains1])
                break
            if (remains1 > remains2):
                tree_word_id.add(id1[-remains1])
                remains1 -= 1
            else:
                tree_word_id.add(id2[-remains2])
                remains2 -= 1
        tree_word_id -= set([entity1.ID, entity2.ID])
        noun_and_verb = map(lambda x: x.LEMMA, filter(lambda x: x.POSTAG.startswith("n") or x.POSTAG.startswith("v"),
                                                      map(lambda id: array[id - 1],
                                                          sorted(tree_word_id, reverse=False))))
        words = list(filter(lambda word: not stopwords.__contains__(word), noun_and_verb))
        result = ''
        for word in words:
            result += word + " "
            # if len(result)>5:
            #     break
        relations_detail.append(result.strip())
    return relations_detail

def saveRelationWord(relation,words):
    result = []
    try:
        with open("../data/relation/" + relation + ".txt", "w") as f:
            for word in words:
                f.write(word+"\n")
    except:
        pass
    return result

if __name__ == '__main__':
    # pprint(relations_en)
    relation_admin = relation_admin_zh

    if relation_admin.relation_word_dic.__len__()==0:
        class_corpus = {}
        with open("../data/train_zh.txt", "r",encoding="UTF-8") as f:
            for line in f.readlines():
                line = line.strip()
                classification = line.split("|")[0].strip()
                sentence = line.split("|")[1].strip()
                if not class_corpus.__contains__(classification):
                    class_corpus[classification] = []
                if (len(sentence) > 0):
                    class_corpus[classification].append(sentence)
        for classification in class_corpus.keys():
            result =  generateRelationWord(class_corpus[classification])
            print(classification,result)
            saveRelationWord(classification,result)
