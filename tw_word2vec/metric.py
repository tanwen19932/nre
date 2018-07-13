import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, recall_score, precision_score


class Metrics(Callback):
    def __init__(self, sentence_vector):
        super().__init__()
        self.sentence_vector = sentence_vector

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        if len(self.validation_data) ==0:
            print("验证集大小为0！")
            return
        val_predict = numpy.asarray(self.model.predict(
            {'sequence_input': self.validation_data[0],
             'posi_input': self.validation_data[1],
             'pos_input': self.validation_data[2]
             }))
        val_preds = self.sentence_vector.prop2index(val_predict)
        val_targ = self.sentence_vector.prop2index(self.validation_data[3])
        # labels = [i for i in range(len(self.sentence_vector.inputer.types))]
        labels = [x for x in self.sentence_vector.inputer.types]
        ## valdidate
        ## [0,0,0,0,0,1]
        ## [0,1,0,0,0,0]
        ## predict
        ## [average0.2,0.1,0,0,0,0.8]
        ## [0,1,0,0,0,0]
        labels.append("others")
        _val_f1 = f1_score(val_targ, val_preds, labels=labels, average="micro")
        _val_recall = recall_score(val_targ, val_preds, labels=labels, average="micro")
        _val_precision = precision_score(val_targ, val_preds, labels=labels, average="micro")
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("f1:",_val_f1 ,"recall",_val_recall,"val_precision:",_val_precision)
        return
