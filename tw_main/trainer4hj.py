from wsgiref import simple_server

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append("/Users/tw/PycharmProjects/nre")

import falcon

from tw_webservice.ws import ReWebService
from tw_word2vec.bilstm_trainer_zh import BiLstmTrainer
from tw_word2vec.inputer import Inputer, Configuration
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
    predict_texts = ["<loc>美国</loc>目前共有2级11艘航空母舰，包括企业级核动力航母1艘，尼米兹级核动力航母10<loc>艘，</loc>全部采用核动力发动机",
                     "<loc>美国</loc>经过多年航空母舰的发<loc>展，</loc>一直以来都是全球拥有最多、排水量和体积最大、舰载机搭载数量最多、作战效率最强大、而且全部使用核动力航空母舰的国家"]
    import json

    print("测试数据", json.dumps(outputer.getDescription(predict_texts), ensure_ascii=False))

    re_service = ReWebService(outputer)
    api = falcon.API()
    api.add_route('/re', re_service)
    httpd = simple_server.make_server('192.168.0.8', 65502, api)
    print("服务开启成功！")
    httpd.serve_forever()
