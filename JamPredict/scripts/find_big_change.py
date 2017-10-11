#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-9-8, 10:05

@Description:

@Update Date: 17-9-8, 10:05
"""

from jampredict.utils import compare

is_mmn = False
month = "all"


def main():
    fn = "48_48_20_LinearInterpolationFixed"
    save_path = "./data/2016/all/" + fn +"_diff.csv"
    compare.find_big_change("all", fn, save_path)


if __name__ == '__main__':
    main()
