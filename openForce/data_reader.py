
# coding: utf-8

# Library required for data handling
import pandas as pd
import numpy as np
import datetime as dt

import glob,os

import warnings

import c3d

class DataReader:
    def __init__(self,relative_data_folder,filename,data_freq=1000,
                 use_filename=True,ext_name='',use_ext_name=False,
                 skiprows=0,column_name=[]):
        self.filename = filename
        self.relative_data_folder = relative_data_folder
        self.data_freq = data_freq
        self.skiprows = skiprows
        self.column_name = column_name
        self.data_type = ''
        if use_filename == True:
            self.filenames = [self.filename]
        else:
            if use_ext_name == True:
                path = self.relative_data_folder + '*' + ext_name
                self.filenames = [os.path.basename(r) for r in glob.glob(path)]
            else:
                warnings.warn('Try to read data, but can not get filenames ...')

        print('There are file name lists: ')
        print(self.filenames)

        print('Detect format type ...')
        self.detect_data_type()
        print('Data type is ',self.data_type)

        print('Try to read data from now ...')

        self.df_list = []
        self.read_data()

        print('Data read finishd.')

    def read_data(self):
        for filename in self.filenames:
            relative_data_path = self.relative_data_folder+filename
            if self.data_type == 'csv':
                if self.column_name == []:
                    force_plate_df = pd.read_csv(relative_data_path, skiprows=self.skiprows)
                else:
                    force_plate_df = pd.read_csv(relative_data_path, skiprows=self.skiprows,
                                                 header=None, names=self.column_name)
                force_plate_end_time =  (len(force_plate_df)-1)/float(self.data_freq)
                force_plate_time = np.linspace(0,force_plate_end_time,len(force_plate_df))
                force_plate_df['time'] = force_plate_time
                self.df_list.append(force_plate_df)
            elif self.data_type == 'c3d':
                reader = c3d.Reader(open(relative_data_path, 'rb'))
                key_label = 'Tonkachi'
                target_label_position = []
                for i,label_name in enumerate(reader.point_labels):
                    if key_label in label_name:
                        target_label_position.append(i)
                        print('Label name', label_name)
                print('Target label positon: ',target_label_position)
                print('')

                points_list = []
                for i,points,analog in reader.read_frames():
                    points_tmp = []
                    for i,point in enumerate(points):
                        if i in target_label_position:
                            label = reader.point_labels[i].strip()
                            label_x = label+'_x'
                            label_y = label+'_y'
                            label_z = label+'_z'
                            points_tmp.append({label_x:point[0],label_y:point[1],label_z:point[2]})
                    tmp = {k: v for dic in points_tmp for k, v in dic.items()}
                    points_list.append(tmp)
                # print(print(np.array(points_list)))
                tmp_df = pd.DataFrame(points_list)
                end_time =  (len(tmp_df)-1)/float(self.data_freq)
                time = np.linspace(0,end_time,len(tmp_df))
                tmp_df['time'] = time
                self.df_list.append(tmp_df)

    def detect_data_type(self):
        filename = self.filenames[0]
        if '.csv' in filename:
            self.data_type = 'csv'
        elif '.txt' in filename:
            self.data_type = 'txt'
        elif '.c3d' in filename:
            self.data_type = 'c3d'
