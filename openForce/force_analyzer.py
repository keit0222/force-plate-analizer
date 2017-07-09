
# coding: utf-8

import numpy as np

# Libraries necessary for visualizing
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import warnings

from scipy.signal import argrelmax,argrelmin

from data_reader import DataReader

class forceAnalyzer(DataReader):
    def __init__(self,relative_data_folder,filename):
        super().__init__(relative_data_folder,filename)

        # Setting at the time of visualizing
        plt.style.use('ggplot')
        font = {'family' : 'mbeiryo'}
        matplotlib.rc('font', **font)



    def moving_filter(self,x,window_size,min_periods):
        return pd.Series(x).rolling(window=window_size, min_periods=min_periods, center=True).mean().values

    def EMA(self,x, alpha):
        return pd.Series(x).ewm(alpha=alpha).mean()

    def getNearestValue(self,array, num):
        """
        概要: リストからある値に最も近い値を返却する関数
        @param list: データ配列
        @param num: 対象値
        @return 対象値に最も近い値
        """
        # リスト要素と対象値の差分を計算し最小値のインデックスを取得
        idx = np.abs(array - num).argmin()
        return array[idx]


    def get_first_landing_point(self):
        x_max_peek = argrelmax(force_plate_data['Fz'].values,order=50)
        x_min_peek = argrelmin(force_plate_data['Fz'].values,order=100)
        # print(x_min_peek,x_max_peek)

        offset_peek_list= []
        for value in x_max_peek[0]:
            if abs(force_plate_data['Fz'].values[value] - force_plate_data['Fz'].values[getNearestValue(x_min_peek[0],value)]) > 100:
                offset_peek_list.append(value)
        # print(offset_peek_list)
        first_landing_point = offset_peek_list[0]
        print('first landing point is ',first_landing_point)


    def export_from_first_landing_point(self):
        force_cutted_df = force_plate_data[first_landing_point:len(force_plate_data)]
        print(force_cutted_df)
        force_cutted_df.plot(y='Fz', figsize=(16,4), alpha=0.5)
        force_cutted_df.to_csv('force_plate_cutted_data_a6.csv')


    def plot(self,analysis_id=0):
        target_area = ['Fx','Fy','Fz','Mx','My','Mz']

        force_plate_data = self.df_list[analysis_id].copy()
        print(force_plate_data)
        column_name = force_plate_data.columns.values

        column_name_tmp = []
        column_name_tmp_array = []
        for target_name in target_area:
            column_name_tmp =  [name for name in column_name if target_name in name]
            column_name_tmp_array.extend(column_name_tmp)
        column_name = column_name_tmp_array
        print(column_name)

        f = plt.figure()
        plt.title('Force plate csv data when liftting up object', color='black')
        force_plate_data.plot(x='time',y=column_name[0:3], figsize=(16,4), alpha=0.5,ax=f.gca())
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        f.subplots_adjust(right=0.8)
        plt.show()
        plt.close()
