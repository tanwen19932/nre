#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : multi_layer.py
# @Author: TW
# @Date  : 2018/1/31
# @Desc  :
from keras.layers import Conv1D, Conv2D


class MultiConv1D(object):
    filter_index = -1
    kernel_index = -1

    def __init__(self, filters: list, kernel_size: list, activation) -> None:
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

    def has_next(self) -> bool:
        if self.filter_index + 1 < len(self.filters):
            return True
        elif self.filter_index + 1 == len(self.filters) and self.kernel_index + 1 < len(self.kernel_size):
            return True
        else:
            return False

    def change2next(self):
        if self.filter_index==-1:
            self.filter_index = 0
        if self.kernel_index + 1 < len(self.kernel_size):
            res = Conv1D(filters=self.filters[self.filter_index], kernel_size=self.kernel_size[self.kernel_index+1],
                         activation=self.activation)
            self.kernel_index += 1
            return res
        elif self.filter_index + 1 < len(self.filters):
            self.filter_index += 1
            res = Conv1D(filters=self.filters[self.filter_index], kernel_size=self.kernel_size[self.kernel_index],
                         activation=self.activation)
            return res
        else:
            return None

    def __iter__(self):
        return self  # 实例本身就是迭代对象，故返回自己

    def __next__(self):
        if not self.has_next():  # 退出循环的条件
            raise StopIteration()
        return self.change2next()  # 返回下一个值

    def __getitem__(self, n):
        count = 0
        for filter in self.filters:
            for kernel in self.kernel_size:
                if (count == n):
                    return Conv1D(filters=filter, kernel_size=kernel,
                                  activation=self.activation)
        return None


class MultiConv2D(object):
    filter_index = 0
    kernel_index = 0
    total_n = 0

    def __init__(self, filters: list, kernel_size: list, activation) -> None:
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

    def has_next(self) -> bool:
        if self.filter_index + 1 < len(self.filters):
            return True
        elif self.filter_index + 1 == len(self.filters) and self.kernel_index + 1 < len(self.kernel_size):
            return True
        else:
            return False

    def change2next(self):
        if self.filter_index == -1:
            self.filter_index = 0
        if self.kernel_index + 1 < len(self.kernel_size):
            res = Conv2D(filters=self.filters[self.filter_index], kernel_size=self.kernel_size[self.kernel_index + 1],
                         activation=self.activation)
            self.kernel_index += 1
            return res
        elif self.filter_index + 1 < len(self.filters):
            self.filter_index += 1
            res = Conv2D(filters=self.filters[self.filter_index], kernel_size=self.kernel_size[self.kernel_index],
                         activation=self.activation)
            return res
        else:
            return None

    def __iter__(self):
        return self  # 实例本身就是迭代对象，故返回自己

    def __next__(self):
        if not self.has_next():  # 退出循环的条件
            raise StopIteration()
        return self.change2next()  # 返回下一个值

    def __getitem__(self, n):
        count = 0
        for filter in self.filters:
            for kernel in self.kernel_size:
                if (count == n):
                    return Conv2D(filters=filter, kernel_size=kernel,
                                  activation=self.activation)
                else: count+=1
        return None


if __name__ == '__main__':
    conv1d_1s = MultiConv1D(filters=[90, 80, 70], kernel_size=[3, 4, 5], activation='sigmoid')
    best_model = None
    count = 0
    for conv1d in conv1d_1s:
        # print("--",conv1d_1s.filter_index,conv1d_1s.kernel_index)
        pass
