from tw_word2vec.inputer import SentencesVector
import numpy as np
import torch
import torch.utils.data as D
from ltl_pytorch.ACNN import ACNN
from ltl_pytorch.ACNN import NovelDistanceLoss
from sklearn.model_selection import KFold


class ACNN_trainer():
    def __init__(self, sentences_vector: SentencesVector):
        self.LR = 0.2
        self.NR = 19
        self.slide_window = 3
        self.num_filters = 150
        self.dropRate = 0.5
        self.num_epochs = 6000
        self.batch_size = 100

        self.sentences_vector = sentences_vector

        self.size_embedding = self.inputer.EMBEDDING_DIM
        self.max_seq_len = sentences_vector.inputer.MAX_NB_WORDS
        self.inputer = self.sentences_vector.inputer
        self.config = self.inputer.config
        self.senVec = self.sentences_vector.embedded_sequences
        self.rela = self.sentences_vector.classifications_vec
        self.e1 = self.sentences_vector.e1Vec
        self.e2 = self.sentences_vector.e2Vec
        self.posVec = self.sentences_vector.position_vec

    def data_unpack(self, cat_data, target):
        N = self.max_seq_len
        list_x = np.split(cat_data.numpy(), [N, N + 1, N + 2], 1)
        bx = torch.from_numpy(list_x[0])
        be1 = torch.from_numpy(list_x[1])
        be2 = torch.from_numpy(list_x[2])
        posVec = torch.from_numpy(list_x[3])
        target = target
        return bx, be1, be2, posVec, target

    # 获取batch批处理
    def get_batches(Xs, ys, batch_size):
        for start in range(0, len(Xs), batch_size):
            end = min(start + batch_size, len(Xs))
            yield Xs[start:end], ys[start:end]

    def train(self):

        model = ACNN(max_len=self.max_seq_len, dim_pos=self.sentences_vector.dim_pos,
                     slide_window=self.slide_window, dim_word=self.size_embedding, types=len(self.inputer.types),
                     num_filters=self.num_filters, dropRate=self.dropRate
                     )
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)
        loss_func = NovelDistanceLoss(self.NR)
        # 把这些先连接起来，分出batch之后再拆开
        input_cat = np.concatenate(
            (self.senVec, self.e1.reshape(-1, 1), self.e2.reshape(-1, 1), self.posVec),
            1)
        # 划分训练集和验证集
        kf = KFold(n_splits=5)
        for epoch_i in range(self.num_epochs):
            acc = 0
            loss = 0
            # train_index, test_index都是一次划分形成的一个集合
            for train_index, val_index in kf.split(input_cat):
                # print("TRAIN:", train_index, "TEST:", test_index)
                train_cat_x, val_cat_x = self.senVec[train_index], self.senVec[val_index]
                train_y, val_y = self.rela[train_index], self.rela[val_index]
                j = 0
                train_datasets = D.TensorDataset(train_cat_x, train_y)
                train_dataloader = D.DataLoader(train_datasets, self.batch_size, True, num_workers=0)
                for cat in train_dataloader:
                    cat_x = cat[0]
                    y = cat[1]
                    x, e1, e2, posVec, y = self.data_unpack(cat_x, y)
                    # 这里应该是调用forward了
                    wo, rela_weight = model(x, e1, e2, posVec)
                    acc += self.prediction(wo, rela_weight, y, self.NR)
                    l = loss_func(wo, rela_weight, y)
                    j += 1
                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
                    loss += l

                val_x = torch.from_numpy(val_x.astype(np.int64))
                val_y = torch.from_numpy(val_y.astype(np.int64))
                val_acc = 0
                ti = 0
                val_datasets = D.TensorDataset(data_tensor=val_cat_x, target_tensor=val_y)
                val_dataloader = D.DataLoader(val_datasets, self.batch_size, True, num_workers=0)
                for (v_x, v_y) in val_dataloader:
                    wo, rel_weight = model(v_x, v_y)
                    val_acc += self.prediction(wo, rel_weight, v_y, self.NR)
                    ti += 1
                print('epoch:', epoch_i, 'acc:', acc / j, '%   loss:', loss.cpu().data.numpy()[0] / j, 'val_acc:',
                      val_acc / ti, '%')
        torch.save(model.state_dict(), self.config.model_file_path)


