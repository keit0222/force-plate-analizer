
# coding: utf-8

import numpy as np

# Libraries necessary for visualizing
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import warnings

from scipy.signal import argrelmax,argrelmin
import matplotlib.font_manager as fm

import copy

from data_reader import DataReader

import os

class forceAnalyzer(DataReader):
    def __init__(self,relative_data_folder,filename):
        super().__init__(relative_data_folder,filename,skiprows=6,
                         column_name=['Fx','Fy','Fz','Mx','My','Mz','Syncro','ExtIn1','ExtIn2'])
        self.extract_syncronized_data()

        self.max_peek=[]
        self.get_max_peek()

        # # フォント一覧
        # fonts = fm.findSystemFonts()
        # # フォントのパスと名前を取得、とりあえず10個表示
        # print([[str(font), fm.FontProperties(fname=font).get_name()] for font in fonts[:20]])

        # Setting at the time of visualizing
        # plt.style.use('ggplot')
        # font = {'family' : 'Yu Gothic'}
        # matplotlib.rc('font', **font)

        self.action_x = 0
        self.action_y = 0
        self.threshold_Fz = 5.0
        self.get_action_point(scale=2.)

    def extract_syncronized_data(self,analysis_id=0):
        df = self.df_list[analysis_id]
        df_copy = df[df.ExtIn1 == 1].copy()
        df_copy['time'] = df_copy['time'].values - df_copy['time'].values[0]
        self.df_list[analysis_id] = df_copy
        self.df_list[analysis_id] = self.df_list[analysis_id].drop(['Syncro','ExtIn1','ExtIn2'], axis=1)

    def moving_filter(self,x,window_size,min_periods):
        return pd.Series(x).rolling(window=window_size, min_periods=min_periods, center=True).mean().values

    def EMA(self,x, alpha):
        return pd.Series(x).ewm(alpha=alpha).mean()

    def get_peek_action_point(self,analysis_id=0):
        return [self.df_list[analysis_id]['action_x'].values[self.max_peek],
                self.df_list[analysis_id]['action_y'].values[self.max_peek]]

    def get_peek_action_point_for_converting(self):
        xs,ys =self.get_peek_action_point()
        points = []
        for x,y in zip(xs,ys):
            points.append([x,y,0])
        return points

    def get_peek_action_point_for_trans(self):
        xs,ys =self.get_peek_action_point()
        points = []
        for x,y in zip(xs,ys):
            points.append([x,y,0.,1.])
        return points

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

    def get_max_peek(self,threshold=5.,analysis_id=0):
        force_plate_data=self.df_list[analysis_id]
        self.max_peek = list(argrelmax(force_plate_data['Fz'].values,order=1000)[0])
        tmp = copy.deepcopy(self.max_peek)
        for i in tmp:
            if force_plate_data['Fz'].values[i] < threshold:
                self.max_peek.remove(i)

    def get_peek_time(self,analysis_id=0):
        return self.df_list[analysis_id]['time'].values[self.max_peek]

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

    def get_action_point(self,analysis_id=0,scale=1.):
        Mx = self.df_list[analysis_id]['Mx'].values*scale
        My = self.df_list[analysis_id]['My'].values*scale
        Fz = self.df_list[analysis_id]['Fz'].values*scale
        tmp_action_x = []
        tmp_action_y = []
        for mx,my,f in zip(Mx,My,Fz):
            if abs(f) > self.threshold_Fz:
                tmp_action_x.append(my/f)
                tmp_action_y.append(mx/f)
            else:
                tmp_action_x.append(-1)
                tmp_action_y.append(-1)
        self.action_x = np.array(tmp_action_x)
        self.action_y = np.array(tmp_action_y)
        self.df_list[analysis_id]['action_x'] = self.action_x
        self.df_list[analysis_id]['action_y'] = self.action_y

    def add_motion_coordinate_action_point(self, simultaneous_trans_matrix,analysis_id=0):
        motion_coordinate_action_point_x = []
        motion_coordinate_action_point_y = []
        motion_coordinate_action_point_z = []
        for x, y in zip(self.action_x, self.action_y):
            if x == -1 or y == -1:
                motion_coordinate_action_point_x.append(0)
                motion_coordinate_action_point_y.append(0)
                motion_coordinate_action_point_z.append(0)
            else:
                arr = np.array([x, y, 0., 1.])
                motion_pos = np.dot(simultaneous_trans_matrix, arr)
                motion_coordinate_action_point_x.append(motion_pos[0])
                motion_coordinate_action_point_y.append(motion_pos[1])
                motion_coordinate_action_point_z.append(motion_pos[2])
        self.df_list[analysis_id]['motion_coordinate_action_point_x'] = motion_coordinate_action_point_x
        self.df_list[analysis_id]['motion_coordinate_action_point_y'] = motion_coordinate_action_point_y
        self.df_list[analysis_id]['motion_coordinate_action_point_z'] = motion_coordinate_action_point_z

    def save_data(self, save_dir, filename, analysis_id=0, update=False):
        if not os.path.isdir(save_dir+'synchro'):
            print('Creating new save folder ...')
            print('Save path : ', save_dir+'synchro')
            os.mkdir(save_dir+'synchro')
        if not os.path.isfile(save_dir+'synchro\\'+filename) or update == True:
            df_copy = self.df_list[analysis_id].copy()
            df_copy = df_copy.set_index('time')
            df_copy.to_csv(save_dir+'synchro\\'+filename)

    def plot_peek_action_point(self):
        xs,ys =self.get_peek_action_point()
        f = plt.figure()
        i = 0
        for x,y in zip(xs,ys):
            plt.plot(x, y, "o", color=cm.spectral(i/10.0))
            i += 1
        f.subplots_adjust(right=0.8)
        plt.show()
        plt.close()


    def plot(self,analysis_id=0):
        target_area = ['Fx','Fy','Fz','Mx','My','Mz']

        force_plate_data = self.df_list[analysis_id].copy()
        # print(force_plate_data)
        column_name = force_plate_data.columns.values

        column_name_tmp = []
        column_name_tmp_array = []
        for target_name in target_area:
            column_name_tmp =  [name for name in column_name if target_name in name]
            column_name_tmp_array.extend(column_name_tmp)
        column_name = column_name_tmp_array
        # print(column_name)

        f = plt.figure()
        plt.title('Force plate csv data when liftting up object', color='black')
        force_plate_data.plot(x='time',y=column_name[0:3], figsize=(16,4), alpha=0.5,ax=f.gca())
        plt.plot(force_plate_data['time'].values[self.max_peek], force_plate_data['Fz'].values[self.max_peek], "ro")
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        f.subplots_adjust(right=0.8)
        plt.show()
        plt.close()

class motionAnalyzer(DataReader):
    def __init__(self,relative_data_folder,filename,key_label):
        super().__init__(relative_data_folder,filename,data_freq=100,key_label_name=key_label)
        self.simple_modify_data()
        self.peek_time = 0

    def simple_modify_data(self):
        for df in self.df_list:
            for i in range(len(df)-1):
                tmp = df.iloc[i+1]
                for j,x in enumerate(tmp[0:9]):
                    if x == 0:
                        df.iloc[i+1,j] = df.iloc[i,j]

    def set_peek_time(self,peek_time):
        self.peek_time = peek_time
        self.get_nearest_time()

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

    def getNearestIndex(self,array, num):
        """
        概要: リストからある値に最も近い値を返却する関数
        @param list: データ配列
        @param num: 対象値
        @return 対象値に最も近い値のインデックス
        """
        # リスト要素と対象値の差分を計算し最小値のインデックスを取得
        return np.abs(array - num).argmin()

    def get_nearest_time(self,analysis_id=0):
        tmp = copy.deepcopy(self.peek_time)
        self.peek_time = []
        for x in tmp:
            self.peek_time.append(self.getNearestIndex(self.df_list[analysis_id]['time'].values ,x))

    def get_peek_points(self, analysis_id=0):
        tmp = self.df_list[analysis_id].iloc[self.peek_time].values
        return tmp[:,:9]

    def get_euclid_distance(self,vec):
        return np.linalg.norm(vec)

    def get_nearest_two_points(self):
        for points in self.get_peek_points():
            each_point = [[points[0:3]],[points[3:6]],[points[6:9]]]
            points_num = [[0,1],[1,2],[2,0]]
            distance = []
            distance.append(np.array(each_point[0])-np.array(each_point[1]))
            distance.append(np.array(each_point[1])-np.array(each_point[2]))
            distance.append(np.array(each_point[2])-np.array(each_point[0]))
            tmp = [100000,-1]
            for i,dis in enumerate(distance):
                tmp_dis = self.get_euclid_distance(dis)
                if  tmp_dis < tmp[0]:
                    tmp = [tmp_dis,i]
            break

        two_points = []
        for points in self.get_peek_points():
            each_point = [[points[0:3]],[points[3:6]],[points[6:9]]]
            two_points.append([each_point[points_num[tmp[1]][0]],each_point[points_num[tmp[1]][1]]])

        return two_points

    def get_middle_point(self,two_points):
        return (np.array(two_points[0])+np.array(two_points[1]))/2

    def get_action_point(self):
        action_points = []
        for points in self.get_nearest_two_points():
            action_points.append(self.get_middle_point(points)[0])
        return action_points

    def get_action_point_for_trans(self):
        action_points = []
        for points in self.get_nearest_two_points():
            tmp = list(self.get_middle_point(points)[0])
            tmp.append(1.0)
            action_points.append(tmp)
        return action_points

    def plot(self,analysis_id=0):
        motion_data = self.df_list[analysis_id].copy()
        column_name = motion_data.columns.values
        # print(column_name[0:len(column_name)-1])

        f = plt.figure()
        plt.title('Motion c3d data', color='black')
        motion_data.plot(x='time',y=column_name[0:len(column_name)-1], figsize=(16,4), alpha=0.5,ax=f.gca())
        if not self.peek_time == 0:
            plt.plot(motion_data['time'].values[self.peek_time], motion_data[column_name[0]].values[self.peek_time], "ro")
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        f.subplots_adjust(right=0.8)
        plt.show()
        plt.close()
