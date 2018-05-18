# -*- coding: utf-8 -*-
import falcon
import sys
from falcon import RequestOptions
from wsgiref import simple_server
import json

from tw_word2vec.lstm_trainer_zh import LstmTrainer
from tw_word2vec.output_zh import Outputer


class ReWebService(object):
    outputer = None
    def __init__(self):
        self.outputer = Outputer(LstmTrainer())

    def on_get(self, req, resp):
        self._handle(req, resp)

    def on_post(self, req, resp):
        self._handle(req, resp)

    def get_request_sentences(self, req):
        sentences = None

        if (req.params.__contains__('sentences')):
            sentences = req.params['sentences']

        elif (req.media.__contains__('sentences')):
            sentences = req.media['sentences']
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
    api = falcon.API()
    captcha = ReWebService()
    api.add_route('/re', captcha)
    httpd = simple_server.make_server('192.168.0.8', 65502, api)
    httpd.serve_forever()
