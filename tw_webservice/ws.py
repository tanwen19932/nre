# -*- coding: utf-8 -*-
from wsgiref import simple_server

import falcon

from tw_segment.en_seg import EnSegmentor
from tw_word2vec.inputer import Configuration, Inputer
from tw_word2vec.lstm_trainer_zh import LstmTrainer
from tw_word2vec.outputer import Outputer
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
    config = Configuration(
        word_segmentor=EnSegmentor(),
        EMBEDDING_DIM=300,
        position_matrix_file_path="../data/posi_matrix.npy",
        word2vec_file_path="../data/needed_word2vec.pkl",
        POS_list_file_path="../data/pos_list.txt",
        types_file_path="../data/relations_en.txt",
        corpus_file_path="../data/train_en.txt",
        model_file_path="../data/model/re_sem_eval_en_model.cnn.hdf5",
    )
    inputer = Inputer(config)
    trainer = Trainer(inputer, LstmTrainer())
    outputer = Outputer(trainer)
    predict_texts = ["所以美军派出<aircraft>“里根”号</aircraft>航母来<loc>南海</loc>增援。",
                     "4月23日,<army>日本海上自卫队</army>“足柄”号和“五月雨”号两艘驱逐舰与<aircraft>“卡尔·文森”号</aircraft>航母打击群在西太平洋举行联合训练。"]
    import json

    print(json.dumps(outputer.getDescription(predict_texts), ensure_ascii=False))
    re_service = ReWebService(outputer)
    api = falcon.API()
    api.add_route('/re', re_service)
    httpd = simple_server.make_server('0.0.0.0', 65502, api)
    httpd.serve_forever()
