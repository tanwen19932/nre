# -*- coding: utf-8 -*-
import falcon
import sys
from falcon import RequestOptions
from wsgiref import simple_server
import json


class ReWebService(object):
    def __init__(self):
        pass

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
            from tw_word2vec import keras_input_zh
            sentences = self.get_request_sentences(req)
            if sentences != None:
                out["result"] = keras_input_zh.getDescription(sentences)
            out["is_ok"] = True
        except Exception as ex:
            print("exception", ex)
            out["is_ok"] = False
            out["exception"] = str(ex)
        resp.status = falcon.HTTP_200
        resp.body = json.dumps(out, ensure_ascii=False)


if __name__ == '__main__':
    # print(captcha.get_output('../jpg/img/lktnjm.jpg'))
    api = falcon.API()
    captcha = ReWebService()
    api.add_route('/re', captcha)
    httpd = simple_server.make_server('127.0.0.1', 8005, api)
    httpd.serve_forever()
