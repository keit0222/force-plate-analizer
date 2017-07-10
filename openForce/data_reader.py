
# coding: utf-8

# Library required for data handling
import pandas as pd
import numpy as np
import datetime as dt

import glob,os

import warnings

class DataReader:
    def __init__(self,relative_data_folder,filename,data_freq=1000,
                 use_filename=True,ext_name='',use_ext_name=False,
                 skiprows=0,column_name=[]):
        self.filename = filename
        self.relative_data_folder = relative_data_folder
        self.data_freq = data_freq
        self.skiprows = skiprows
        self.column_name = column_name
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
        print('Try to read data from now ...')

        self.df_list = []
        self.read_data()

        print('Data read finishd.')
        print(self.df_list)

    def read_data(self):
        for filename in self.filenames:
            relative_data_path = self.relative_data_folder+filename
            if self.column_name == []:
                force_plate_df = pd.read_csv(relative_data_path, skiprows=self.skiprows)
            else:
                force_plate_df = pd.read_csv(relative_data_path, skiprows=self.skiprows,
                                             header=None, names=self.column_name)
            force_plate_end_time =  (len(force_plate_df)-1)/float(self.data_freq)
            force_plate_time = np.linspace(0,force_plate_end_time,len(force_plate_df))
            force_plate_df['time'] = force_plate_time
            self.df_list.append(force_plate_df)
