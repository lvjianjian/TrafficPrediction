#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-7-26, 14:59

@Description:

@Update Date: 17-7-26, 14:59
"""

import urllib
import urllib2
import re
from bs4 import BeautifulSoup


def getPage(month):
    """

    :param month: yyyyMM
    :return:
    """
    url = "http://lishi.tianqi.com/beijing/{}.html".format(month)
    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    return response.read().decode('gbk')


def parse(content):
    soup = BeautifulSoup(content, 'lxml')
    # print soup.prettify()
    infos = soup.find(class_="tqtongji2")
    list = infos.find_all('ul')
    r = []
    for v in list:
        ahref = v.find('a')
        if ahref is not None:
            lis = v.find_all('li')
            date = str(lis[0].text.replace("-", ""))
            maxT = str(lis[1].text)
            minT = str(lis[2].text)
            weather = str(lis[3].text.encode("utf-8"))
            ws = str(lis[5].text.encode("utf-8"))
            r.append(",".join((date, maxT, minT, weather, ws)))
    return r


if __name__ == '__main__':
    import Paramater
    with open(Paramater.DATAPATH+"BJ_WEATHER.csv","w") as f:
        for i in range(1, 13):
            month = "2016" + "%02d" % i
            l = parse(getPage(month))
            for v in l:
                f.write(v+"\n")
        f.flush()