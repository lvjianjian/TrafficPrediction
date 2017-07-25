#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-7-21, 15:25

@Description:

@Update Date: 17-7-21, 15:25
"""

def entropy(Y):
    pass

class BaseLine(object):
    def __init__(self, maxC, maxD, maxW, minSupport, minConfidence):
        super(BaseLine, self).__init__()
        self._root = None
        self._maxC = maxC
        self._maxD = maxD
        self._maxW = maxW
        self._minSupport = minSupport
        self._minConfidence = minConfidence

    def fit(self, X, Y, c, d, w):
        print X.shape
        print Y.shape

    def predict(self, X):
        pass

    def _build(self):
        pass


class Node(object):
    def __init__(self, X, Y, nc, nd, nw, es, minSupport, minConfidence):
        super(Node, self).__init__()
        self._X = X
        self._Y = Y
        self._nc = nc
        self._nd = nd
        self._nw = nw
        self._es = es
        self._minSupport = minSupport
        self._minConfidence = minConfidence
        self._choose = -1
        self._childs = {}



    def _getBest(self):
        pass


