# -*- coding: utf-8 -*-
from wsgiref import simple_server

import falcon

from tw_segment.en_seg import EnSegmentor
from tw_segment.jieba_seg import JieBaTokenizer
from tw_word2vec.bilstm_trainer_zh import BiLstmTrainer
from tw_word2vec.cnn_trainer_zh import CnnTrainer
from tw_word2vec.inputer import Configuration, Inputer
from tw_word2vec.lstm_trainer_zh import LstmTrainer
from tw_word2vec.outputer import Outputer
from tw_word2vec.sem_eval_08 import CnnTrainerEn
from tw_word2vec.trainer import Trainer
import json


class ReWebService(object):
    def __init__(self, outputer: Outputer):
        self.outputer = outputer

    def on_get(self, req, resp):
        self._handle(req, resp)

    def on_post(self, req, resp):
        self._handle(req, resp)

    def get_request_sentences(self, req):
        sentences = None
        req_body = req.bounded_stream.read()
        json_data = json.loads(req_body.decode('utf8'))
        sentences = json_data['sentences']
        return sentences

    def _handle(self, req, resp):
        out = dict()
        try:
            sentences = self.get_request_sentences(req)
            if sentences != None:
                out["result"] = self.outputer.getDescription(sentences)
            out["is_ok"] = True
        except Exception as ex:
            print("exception", ex)
            out["is_ok"] = False
            out["exception"] = str(ex)
        resp.status = falcon.HTTP_200
        resp.body = json.dumps(out, ensure_ascii=False)


if __name__ == '__main__':
    # 导入库
    import argparse

    # 1. 定义命令行解析器对象
    parser = argparse.ArgumentParser(description='基于深度学习的基本关系识别程序')

    # 2. 添加命令行参数
    parser.add_argument('-lang', choices=['en', 'zh'], help="语言选项 zh和en", default="en")
    parser.add_argument('-pos', help="位置信息向量路径",
                        default="./data/posi_matrix.npy")
    parser.add_argument('-POS', help="实体类别列表",
                        default="./data/sem_eval_2010_task8/pos_list.txt")
    parser.add_argument('-type', help="关系类别列表",
                        default="./data/sem_eval_2010_task8/relations_en.txt")
    parser.add_argument('-train', help="训练集信息",
                        default="./data/sem_eval_2010_task8/train_en.txt")
    parser.add_argument('-model', type=str, help="模型存储位置信息",
                        default="./data/model/re_sem_eval_en_model.lstm.hdf5")
    parser.add_argument('-log', type=str, help="模型训练日志存储位置信息",
                        default="./data/model/logs")
    parser.add_argument('-network', choices=['CNN', 'LSTM', 'BiLSTM'], help="模型选项 CNN LSTM BiLSTM", default="LSTM")
    parser.add_argument('-epoch', type=int, help="执行epoch次数", default=100)
    parser.add_argument('-service', type=bool, choices=[True, False], help="是否发布服务", default=False)
    # 3. 从命令行中结构化解析参数
    args = parser.parse_args()
    print(args)
    word_segmentor = EnSegmentor()
    embedding_dim = 300
    word2vec_file_path = "./data/needed_word2vec.pkl"
    if args.lang == 'zh':
        word_segmentor = JieBaTokenizer()
        embedding_dim = 64
        word2vec_file_path = "./data/needed_zh_word2vec.pkl"
    import os

    try:
        dirname = os.path.dirname(args.model)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    except Exception as e:
        print(e)

    config = Configuration(
        word_segmentor=word_segmentor,
        EMBEDDING_DIM=embedding_dim,
        position_matrix_file_path=args.pos,
        word2vec_file_path=word2vec_file_path,
        POS_list_file_path=args.POS,
        types_file_path=args.type,
        corpus_file_path=args.train,
        model_file_path=args.model,
        log_file_path=args.log,
        epoch=args.epoch,
    )
    inputer = Inputer(config)
    network = LstmTrainer()
    if args.network == 'CNN':
        network = CnnTrainer()
        if args.lang == 'en':
            network = CnnTrainerEn()
    if args.network == 'BiLSTM':
        network = BiLstmTrainer()
    trainer = Trainer(inputer, network)
    outputer = Outputer(trainer)
    predict_texts = ["所以美军派出<aircraft>“里根”号</aircraft>航母来<loc>南海</loc>增援。",
                     "4月23日,<army>日本海上自卫队</army>“足柄”号和“五月雨”号两艘驱逐舰与<aircraft>“卡尔·文森”号</aircraft>航母打击群在西太平洋举行联合训练。"]
    import json

    print(json.dumps(outputer.getDescription(predict_texts), ensure_ascii=False))
    if args.service:
        re_service = ReWebService(outputer)
        api = falcon.API()
        api.add_route('/re', re_service)
        httpd = simple_server.make_server('0.0.0.0', 65502, api)
        print("服务生效中")
        httpd.serve_forever()
