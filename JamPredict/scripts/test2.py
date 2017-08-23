#!/usr/bin/python
#-*- coding:utf-8 -*-
from jampredict.utils import Paramater
import theano.tensor as T
import theano
from keras.layers.recurrent import SimpleRNN

class A(object):
    def test1(self):
        return getattr(self,"a",False)

    def test2(self):
        self.a = True

if __name__ == '__main__':
    import pickle

    # pkl_file = open(Paramater.PROJECTPATH+"RET/c3.p1.t1.resunit6.lr0.0002.External.MMN.history.pkl", 'rb')
    #
    # history = pickle.load(pkl_file)
    # print(history)
    # T.nnet.softmax()
    a = A()
    print a.test1()
    a.test2()
    print a.test1()

