from torch import nn
import numpy as np
import torch
import torch.nn.functional as F

class ACNN(nn.Module):
    def one_hot(indices, depth, on_value=1, off_value=0):
        np_ids = np.array(indices.cpu().data.numpy()).astype(int)
        if len(np_ids.shape) == 2:
            encoding = np.zeros([np_ids.shape[0], np_ids.shape[1], depth], dtype=int)
            added = encoding + off_value
            for i in range(np_ids.shape[0]):
                for j in range(np_ids.shape[1]):
                    added[i, j, np_ids[i, j]] = on_value
            return torch.FloatTensor(added.astype(float))
        if len(np_ids.shape) == 1:
            encoding = np.zeros([np_ids.shape[0], depth], dtype=int)
            added = encoding + off_value
            for i in range(np_ids.shape[0]):
                added[i, np_ids[i]] = on_value
            return torch.FloatTensor(added.astype(float))

    def __init__(self, max_len, dim_pos,
                 slide_window, dim_word, types,
                 num_filters, dropRate):
        super(ACNN, self).__init__()
        self.dim_pos = dim_pos
        self.dim = self.dim_word + 2 * self.dim_pos
        self.num_filters = num_filters
        self.dim_word = dim_word
        self.keep_prob = dropRate
        # 滑动窗口只支持奇数个
        self.slide_window = slide_window
        self.k = (self.slide_window - 1) // 2
        self.p = (self.k - 1) // 2
        self.max_len = max_len
        self.kd = self.d * self.k
        self.types = types

        self.pad = nn.ConstantPad2d((0, 0, self.k, self.k), 0)
        self.y_embedding = nn.Embedding(self.nr, self.num_filters)

        self.dropout = nn.Dropout(self.keep_prob)
        # self.conv = nn.Conv2d(1, self.dc, (self.k, self.kd), (1, self.kd), (self.p, 0), bias=True)
        # 第一个元组是卷积核尺寸，第二个元组是卷积步长
        self.conv = nn.Conv2d(1, self.num_filters, (1, self.dim), (1, self.dim), bias=True)  # renewed
        self.tanh = nn.Tanh()
        self.U = nn.Parameter(torch.randn(self.num_filters, self.num_filters))
        self.We1 = nn.Parameter(torch.randn(self.dim_word, self.dim_word))
        self.We2 = nn.Parameter(torch.randn(self.dim_word, self.dim_word))
        self.max_pool = nn.MaxPool2d((1, self.types), (1, self.types))
        self.softmax = nn.Softmax()

    def window_cat(self, x_concat):
        # view方法重新塑造一个
        s = x_concat.data.size()
        # 弄出这个1的原因可能是，进行pad的时候是按句子进行的，每次的输入其实只是(s[1],s[2]),而那个pad层只能....
        px = self.pad(x_concat.view(s[0], 1, s[1], s[2])).view(s[0], s[1] + 2 * self.dim_pos, s[2])
        t_px = torch.index_select(px, 1, torch.LongTensor(range(s[1])))
        m_px = torch.index_select(px, 1, torch.LongTensor(range(1, s[1] + 1)))
        b_px = torch.index_select(px, 1, torch.LongTensor(range(2, s[1] + 2)))
        return torch.cat([t_px, m_px, b_px], 2)

    def input_attention(self, x, e1, e2, posVec, is_training=True):
        bz = x.data.size()[0]
        x_embed = x
        e1_embed = e1
        e2_embed = e2
        x_concat = torch.cat((x_embed, posVec), 2)
        w_concat = self.window_cat(x_concat)
        if is_training:
            w_concat = self.dropout(w_concat)
        #     以下是一个全连接层
        #     repeat是按照各个方向分别重复bz,1,1倍，形成新的参数
        W1 = self.We1.view(1, self.dim_word, self.dim_word).repeat(bz, 1, 1)
        W2 = self.We2.view(1, self.dim_word, self.dim_word).repeat(bz, 1, 1)
        #bmm运算： bz * ml * dw 和 bz * dw * dw   ====>>>>>    bz * ml * dw
        W1x = torch.bmm(x_embed, W1)
        W2x = torch.bmm(x_embed, W2)
        A1 = torch.bmm(W1x, e1_embed.view(bz, self.dim_word, 1))  # (bz, ml, 1)
        A2 = torch.bmm(W2x, e2_embed.view(bz, self.dim_word, 1))
        A1 = A1.view(bz, self.max_len)
        A2 = A2.view(bz, self.max_len)
        alpha1 = self.softmax(A1)
        alpha2 = self.softmax(A2)
        alpha = torch.div(torch.add(alpha1, alpha2), 2)
        alpha = alpha.view(bz, self.max_len, 1).repeat(1, 1, self.dim_wordPos)
        return torch.mul(w_concat, alpha)

    def new_convolution(self, R):
        s = R.data.size()  # bz, ml, slide_window * dim_wordPos
        R = self.conv(R.view(s[0], 1, s[1], s[2]))  # bz, num_filters, ml, 1
        R = self.tanh(R)  # added
        R_star = R.view(s[0], self.num_filters, s[1])
        return R_star  # bz, num_filters, ml

    def attentive_pooling(self, convOut):
        rela_weight = self.y_embedding.weight
        bz = convOut.data.size()[0]
        b_U = self.U.view(1, self.num_filters, self.types).repeat(bz, 1, 1)
        b_rel_w = rela_weight.view(1, self.nr, self.dc).repeat(bz, 1, 1)
        # G是学习每一个单词对最后关系的影响
        G = torch.bmm(convOut.transpose(2, 1), b_U)  # (bz, ml, num_filters)
        G = torch.bmm(G, b_rel_w)  # (bz, ml, types)
        AP = F.softmax(G)
        AP = AP.view(bz, self.max_len, self.types)
        wo = torch.bmm(convOut, AP)  # bz, types, types
        wo = self.max_pool(wo.view(bz, 1, self.dc, self.dc))
        return wo.view(bz, 1, self.dc).view(bz, self.dc), rela_weight

    def forward(self, x, e1, e2, posVec, is_training=True):
        attentionOut = self.input_attention(x, e1, e2, posVec, is_training)
        covOut = self.new_convolution(attentionOut)
        wo, rela_weight = self.attentive_pooling(covOut)
        wo = F.relu(wo)
        # wo是池化后的输出张量，rel_weight是关系embed层的权重
        return wo, rela_weight


class NovelDistanceLoss(nn.Module):
    def __init__(self, nr, margin=1):
        # super是把self转变成noveldistanceloss的父类
        super(NovelDistanceLoss, self).__init__()
        self.nr = nr
        self.margin = margin

    def forward(self, wo, rela_weight, in_y):
        # 对输出正则化
        wo_norm = F.normalize(wo)  # (bz, dc)

        bz = wo_norm.data.size()[0]
        dc = wo_norm.data.size()[1]
        # -1是从其他位置推断出来的
        wo_norm_tile = wo_norm.view(-1, 1, dc).repeat(1, self.nr, 1)  # (bz, nr, dc)
        # 这个是各个关系的表征向量
        batched_rel_w = F.normalize(rela_weight).view(1, self.nr, dc).repeat(bz, 1, 1)
        # N2正则，dim=2是在第二维上保持不变，输出还是第二维的个数，但是最后还是会squeeze
        all_distance = torch.norm(wo_norm_tile - batched_rel_w, 2, 2)  # (bz, nr, 1)
        mask = self.one_hot(in_y, self.nr, 1000, 0)  # (bz, nr)
        masked_y = torch.add(all_distance.view(bz, self.nr), mask)
        neg_y = torch.min(masked_y, dim=1)[1]  # (bz,)
        neg_y = torch.mm(self.one_hot(neg_y, self.nr), rela_weight)  # (bz, nr)*(nr, dc) => (bz, dc)
        pos_y = torch.mm(self.one_hot(in_y, self.nr), rela_weight)
        neg_distance = torch.norm(wo_norm - F.normalize(neg_y), 2, 1)
        pos_distance = torch.norm(wo_norm - F.normalize(pos_y), 2, 1)
        loss = torch.mean(pos_distance + self.margin - neg_distance)
        return loss