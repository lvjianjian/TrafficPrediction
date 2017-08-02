#!/usr/bin/python
#-*- coding:utf-8 -*-
from jampredict.utils import Paramater

if __name__ == '__main__':
    import pickle

    pkl_file = open(Paramater.PROJECTPATH+"RET/c3.p1.t1.resunit6.lr0.0002.External.MMN.history.pkl", 'rb')

    history = pickle.load(pkl_file)
    print(history)