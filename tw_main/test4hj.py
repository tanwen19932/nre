import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append("/Users/tw/PycharmProjects/nre")
from tw_word2vec.bilstm_trainer_zh import BiLstmTrainer
from tw_word2vec.inputer import Configuration, Inputer
from tw_word2vec.outputer import Outputer
from tw_word2vec.trainer import Trainer

if __name__ == '__main__':
    config = Configuration(
        position_matrix_file_path="../data/posi_matrix.npy",
        word2vec_file_path="../data1/news_12g_baidubaike_20g_novel_90g_embedding_64.bin",
        POS_list_file_path="../data/relation_hj/pos_list.txt",
        types_file_path="../data/relation_hj/relations_zh.txt",
        corpus_file_path="../data/relation_hj/train_zh.txt",
        model_file_path="../data/model/re_hj_zh_model.bilstm.hdf5",
    )
    inputer = Inputer(config)
    trainer = Trainer(inputer, BiLstmTrainer())
    outputer = Outputer(trainer)

    print("读取模型完成！ 开始准备测试：")
    import time;  # 引入time模块

    while True:

        try:
            file_name = input("请输入文件名：")
            test_sources = []
            test_types = []
            predict_types = []

            with open(file_name, "r", encoding="UTF-8") as f:
                for line in f.readlines():
                    try:
                        test_sources.append(line.split("|")[1])
                        test_types.append(line.split("|")[0])
                    except:
                        print(line)
            begin_time = time.time()

            results = outputer.getDescription(test_sources)
            end_time = time.time()
            elapse_time = (end_time - begin_time) / 1000
            print("一共", len(test_sources), "条数据" "耗时：", elapse_time, " 平均耗时", elapse_time / len(test_sources))
            print("下面计算准确率：")
            right = 0
            wrong_index = []

            for i in range(len(results)):
                result = results[i]
                isRight = False
                if result["predict_type"] == test_types[i]:
                    right += 1
                    isRight = True
                else:
                    wrong_index.append(i)
                print("句子： ", result["sentence"].replace("\n", ""))
                print("实体1： ", result["e1"], "实体1类型： ", result["e1_type"])
                print("实体2： ", result["e2"], "实体2类型： ", result["e2_type"])
                print("实际关系： ", test_types[i])
                print("预测关系： ", result["predict_type"])
                print("预测正确？", isRight)
                print()

            print("正确率：", float(right) / len(results))
        except:
            pass


