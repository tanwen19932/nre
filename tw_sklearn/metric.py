#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : metric.py
# @Author: TW
# @Date  : 2018/5/18
# @Desc  :
from sklearn import metrics


def MF( golds, preds):
    # corrects = float(sum(preds == golds))
    label = [i for i in range(1, 51)]
    accuracy = metrics.accuracy_score(golds, preds)
    micro_p = metrics.precision_score(golds, preds, labels=label, average="micro")
    micro_r = metrics.recall_score(golds, preds, labels=label, average="micro")
    micro_f = metrics.f1_score(golds, preds, labels=label, average="micro")
    macro_p = metrics.precision_score(golds, preds, labels=label, average="macro")
    macro_r = metrics.recall_score(golds, preds, labels=label, average="macro")
    macro_f = metrics.f1_score(golds, preds, labels=label, average="macro")
    p_list = metrics.precision_score(golds, preds, labels=label, average=None)
    r_list = metrics.recall_score(golds, preds, labels=label, average=None)
    f_list = metrics.f1_score(golds, preds, labels=label, average=None)
    return [micro_p, micro_r, micro_f, macro_p, macro_r, macro_f]


if __name__ == '__main__':
    golds = [1,3,4,2,3,5,-1,1]
    preds = [1,3,4,20,3,6,1,-1]
    print(MF(golds, preds))