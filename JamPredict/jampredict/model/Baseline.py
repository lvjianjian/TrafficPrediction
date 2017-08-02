#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-7-21, 15:25

@Description:

@Update Date: 17-7-21, 15:25
"""
import numpy as np


def entropy(Y, labels):
    unique = labels
    count = np.ndarray(shape=unique.shape[0], dtype=float)
    for i, x in enumerate(unique):
        count[i] = (np.sum(Y == x))
    sum = Y.shape[0]
    count = count / sum
    count = count[count != 0]
    return -np.sum(np.log2(count) * count)


def gain(X, Y, labels, choose):
    old_ent = entropy(Y, labels)
    values = np.unique(X[:, choose])
    sum = 0
    all = X.shape[0]
    for value in values:
        _x_size = X[X[:, choose] == value].shape[0]
        _y = Y[X[:, choose] == value]
        ent = entropy(_y, labels)
        sum += float(_x_size) / all * ent
    return old_ent - sum


def maxConfidence(Y, labels):
    count = np.ndarray(shape=labels.shape[0], dtype=float)
    for i, x in enumerate(labels):
        count[i] = (np.sum(Y == x))
    sum = Y.shape[0]
    count = count / sum
    return np.max(count)


def maxConfidenceLabel(Y, labels):
    count = np.ndarray(shape=labels.shape[0], dtype=float)
    for i, x in enumerate(labels):
        count[i] = (np.sum(Y == x))
    sum = Y.shape[0]
    count = count / sum
    return labels[np.argmax(count)]


class BaseLine(object):
    """
    仅针对分类任务
    """
    def __init__(self, maxC, maxD, maxW, minSupport, minConfidence):
        super(BaseLine, self).__init__()
        self._root = None
        self._maxC = maxC
        self._maxD = maxD
        self._maxW = maxW
        self._minSupport = minSupport
        self._minConfidence = minConfidence

    def fit(self, X, Y, c, d, w):
        # assert (c + d + w) >= X.shape[1]
        self._build(X, Y, c, d, w)

    def predict(self, X):
        predicts = np.ndarray(shape=X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            predicts[i] = self._predict(X[i])
        return predicts

    def _predict(self, x):
        cn = self._root
        childs = cn._childs
        while (childs is not None):
            choose = cn._choose
            keys = childs.keys()
            if x[choose] in keys:
                cn = childs[x[choose]]
                childs = cn._childs
            else:  # 选择最近的
                choose_value = keys[np.argmin(np.abs(np.array(keys) - x[choose]))]
                cn = childs[choose_value]
                childs = cn._childs
        return cn._label

    def _build(self, X, Y, c, d, w):
        self._all_labels = np.unique(Y)
        self._c = c
        self._d = d
        self._w = w
        es = []
        for i in range(c + d + w, X.shape[1]):
            es.append(i)
        self._root = Node(X, Y, 0, 0, 0, es, self)


class Node(object):
    def __init__(self, X, Y, nc, nd, nw, es, bl):
        super(Node, self).__init__()
        self._X = X
        self._Y = Y
        self._nc = nc
        self._nd = nd
        self._nw = nw
        self._es = es
        self._choose = -1
        self._childs = None
        self._bl = bl
        self._label = -1
        self._build()

    def _build(self):
        if maxConfidence(self._Y, self._bl._all_labels) > self._bl._minConfidence:  # 叶子节点
            self._label = maxConfidenceLabel(self._Y, self._bl._all_labels)
        else:
            if (self._Y.shape[0] > self._bl._minSupport):  # 可继续分裂
                best_choose = self._getBestAttr()
                if (best_choose == -1):
                    self._label = maxConfidenceLabel(self._Y, self._bl._all_labels)
                else:
                    self._childs = {}
                    self._choose = best_choose
                    # print self._choose
                    values = np.unique(self._X[:, self._choose])
                    if best_choose < self._bl._c:
                        self._nc += 1
                    elif best_choose < self._bl._c + self._bl._d:
                        self._nd += 1
                    elif best_choose < self._bl._c + self._bl._d + self._bl._w:
                        self._nw += 1
                    else:
                        self._es.remove(best_choose)
                    # print self._es
                    for value in values:
                        self._childs[value] = Node(self._X[self._X[:, best_choose] == value],
                                                   self._Y[self._X[:, best_choose] == value],
                                                   self._nc, self._nd, self._nw, self._es, self._bl)


            else:  # 不可继续分裂
                self._label = maxConfidenceLabel(self._Y, self._bl._all_labels)

        self._Y = None
        self._X = None

    def _getBestAttr(self):
        chooses = np.ndarray(shape=3 + len(self._es), dtype=int)
        gains = np.ndarray(shape=3 + len(self._es))

        # c
        if (self._nc < self._bl._maxC and self._nc < self._bl._c):
            gains[0] = gain(self._X, self._Y, self._bl._all_labels, choose=self._nc)
        else:
            gains[0] = -1
        chooses[0] = self._nc

        # d
        if (self._nd < self._bl._maxD and self._nd < self._bl._d):
            gains[1] = gain(self._X, self._Y, self._bl._all_labels, choose=self._bl._c + self._nd)
        else:
            gains[1] = -1
        chooses[1] = self._bl._c + self._nd

        # w
        if (self._nw < self._bl._maxW and self._nw < self._bl._w):
            gains[2] = gain(self._X, self._Y, self._bl._all_labels, choose=self._bl._c + self._bl._d + self._nw)
        else:
            gains[2] = -1
        chooses[2] = self._bl._c + self._bl._d + self._nw
        # es
        for i in range(len(self._es)):
            gains[3 + i] = gain(self._X, self._Y, self._bl._all_labels, choose=self._es[i])
            chooses[3 + i] = self._es[i]
        # print gains
        # print chooses
        if (np.max(gains) < 0):
            final_choose = -1
        else:
            final_choose = chooses[np.argmax(gains)]
        return final_choose


if __name__ == '__main__':
    arange = np.arange(10)
    arange[2] = 4
    arange[8] = 4
    arange[6] = 1
    arange[7] = 1
    arange[8] = 1
    print arange
    print entropy(arange, np.arange(10))

    # print maxConfidence(arange, np.unique(arange))
    # print maxConfidenceLabel(arange, np.unique(arange))
    # list = [2, 4, 6]
    # list.remove(2)
    # print list
    childs = {}
    childs[1] = "a"
    childs[3] = "n"
    keys = childs.keys()
    choose = keys[np.argmin(np.abs(np.array(keys) - 5))]
    print keys
    print choose
