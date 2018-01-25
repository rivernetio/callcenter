# coding:utf-8
import pandas as pd
import numpy as np
from time import time


class AI(object):
    def __init__(self, filename="F:/data/20180105.csv", code="gbk"):
        pd.options.display.float_format = '{:.0f}'.format
        pd.set_option('display.max_columns', 100)
        self.df = pd.read_csv(filename, encoding=code)
        self.data_clean()

    def data_clean(self):

        # filter dailed out & 95588 & area_code + 95588
        self.df = self.df[(self.df['CALL_TYPE'] == 1) & \
                          (self.df['CALLING_NUM'] / 1000000 >= 1) & (
                              (self.df['CALLING_NUM'] % 100000 != 95588) | (self.df['CALLING_NUM'] / 100000000 >= 1))]

        # filter test numbers
        self.df = self.df[self.df['CALLING_NUM'].isin([18210261957, 18210261795, 17343195721, \
                                                       17718332497, 13263105230, 13263103918]) == False]

        # filter IN_TRK_CODE
        self.df = self.df[self.df['IN_TRK_CODE'].isin([4860, 4881, 4815, 4816, 4834, 4873, 4878, 4899, 4820, 4862, \
                                                       4849, 4850, 4828, 4827, 4858, 4859, 4840, 4829, 4848, 4830, \
                                                       4831, 4832, 4833, 4847, 4854, 4705, 4706, 4707, 4708, 4709, \
                                                       4710, 4711, 4712, 4713, 4714, 4855, 4715, 4716, 4717, 4718, \
                                                       4719, 4720, 4721, 4722, 4723, 4724, 4856, 4725, 4726, 4727, \
                                                       4728, 4729, 4730, 4731, 4732, 4733, 4857, 4700, 4701, 4702, \
                                                       4703, 4704, 4872, 4804, 4894, 4801, 4898, 4821, 4805, 4897, \
                                                       4806, 4802, 4892, 4893, 4890, 4861, 4870, 4826, 4885, 4886, \
                                                       4810, 4811, 4852, 4853, 4880, 4879, 4807, 4823, 4876, 4877, \
                                                       4808, 4803, 4812])]

        # df.shape

    @staticmethod
    def count(x):
        return len(np.unique(x))

    # MOST_DURATION_NUM
    @staticmethod
    def most_appearance_num(x):
        return x.value_counts().nlargest(1).values[0]

    # MOST_DURATION
    @staticmethod
    def most_appearance(x):
        return x.value_counts().nlargest(1).index[0]

    # 时间转成分钟
    @staticmethod
    def get_minute(dt):
        dt = str(dt)
        dt_len = len(dt)
        if dt_len < 3:
            dt_min = int(dt)
            dt_hour = 0
        else:
            dt_min = int(dt[-2:])
            dt_hour = int(dt[:-2])
        return dt_hour * 60 + dt_min

    def get_interval(self, x, y):
        return abs(self.get_minute(y) - self.get_minute(x))

    # 获得通话间隔列表
    def interval(self, x):
        x_1 = x[:-1]
        x_2 = x[1:]
        intervals = map(self.get_interval, x_1, x_2)
        return intervals

    # 获取间隔中位数
    def interval_median(self, x):
        if len(x) > 1:
            intervals = np.array(self.interval(x))
            return np.median(np.array(intervals))
        return 60 * 24

    # 获取间隔标准差
    def interval_std(self, x):
        if len(x) > 1:
            intervals = np.array(self.interval(x))
            return np.std(np.array(intervals))
        return np.std(np.array([60 * 24]))

    def main(self, filename):
        # df = df[:10000]
        # TRK_NUM
        # self.df = self.df[:1000]
        result = pd.pivot_table(self.df, index=["CALLING_NUM"], values=["DURATION", "IN_TRK_CODE", "CALLING_TIME"],
                                aggfunc={"DURATION": [np.median, np.sum, np.std, self.most_appearance,
                                                      self.most_appearance_num, len],
                                         "IN_TRK_CODE": [self.count],
                                         "CALLING_TIME": [self.interval_median, self.interval_std]}, fill_value=0)

        result.to_csv('result/%s_result.csv' % filename, index=True)

        # print result.isnull().sum()


if __name__ == "__main__":
    for i in xrange(6, 18):
        start = time()
        print "------------------------------------"
        if i < 10:
            mark = "0"
        else:
            mark = ""
        ai = AI(filename="F:/data/201801/201801" + mark + str(i) + ".csv", code="gbk")
        ai.main("201801" + mark + str(i))
        print "执行文件201801%s%s.csv耗时:" % (mark, str(i)), time() - start
