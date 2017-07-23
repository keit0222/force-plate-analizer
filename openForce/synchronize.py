
# coding : utf-8

import force_analyzer as fa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from coordinate_transform import *

# visualize
from mpl_toolkits.mplot3d import Axes3D

# check whether file exist or not
import os
import glob

# serialize and de-serialize library
import pickle

class Synchro():
    def __init__(self, force_data_dir='',force_file_ext='csv',exception_word='calib'):
        self.force_data_dir = force_data_dir
        self.load_trans_info()

        path = self.force_data_dir + '*.' + force_file_ext
        self.filenames = [os.path.basename(r) for r in glob.glob(path) if not exception_word in os.path.basename(r)]
        print(self.filenames)

        for filename in self.filenames:
            print(force_data_dir+filename.split('.')[0]+'.pkl')
            if os.path.isfile(force_data_dir+filename.split('.')[0]+'.pkl'):
                with open(force_data_dir+filename.split('.')[0]+'.pkl', mode='rb') as f:
                    self.force_cls = pickle.load(f)
            else:
                self.force_cls = fa.forceAnalyzer(force_data_dir, filename)
                with open(force_data_dir+filename.split('.')[0]+'.pkl', mode='wb') as f:
                    pickle.dump(self.force_cls, f)
            self.force_cls.add_motion_coordinate_action_point(self.mat)
            self.force_cls.save_data(self.force_data_dir, filename, update=True)

    def load_trans_info(self):
        if os.path.isfile(self.force_data_dir+'trans_mat'+'.pkl'):
            with open(self.force_data_dir+'trans_mat'+'.pkl', mode='rb') as f:
                self.mat = pickle.load(f)
                print('loaded trans matrix...')

    def get_trans(self):
        self.force_plate_action_points = np.array(self.force_cls.get_peek_action_point_for_trans()).T
        self.motion_action_points = np.array(self.motion_cls.get_action_point_for_trans()).T

        # motion_action_points = M*force_plate_action_points
        # M : 一次変換行列
        # M = motion_action_points*pinv(force_plate_action_points)
        return np.dot(self.motion_action_points,np.linalg.pinv(self.force_plate_action_points))

    def trans_check(self):
        true_motion_position = self.motion_action_points.T
        estimate_position = []
        for point in self.force_plate_action_points.T:
            estimate_position.append(np.dot(self.mat,point.T))

        point_diff = []
        for true_value, estimation_value in zip(true_motion_position,estimate_position):
            # print('True value: ',true_value, ' Estimation value: ',estimation_value)
            point_diff.append(np.array(true_value)-np.array(estimation_value))
        return point_diff

    def get_trans_position(self):
        estimate_position = []
        for point in self.force_plate_action_points.T:
            estimate_position.append(np.dot(self.mat,point.T))
        return estimate_position

    def evaluate_trans(self):
        diff = self.trans_check()
        for point_diff in diff:
            self.diff_norm.append(np.linalg.norm(self.meter2milli(point_diff)))
        # print(diff_norm)
        print('######### Evaluation result ##########')
        print('Error mean: ',np.mean(self.diff_norm),' [mm]')
        print('Error max: ',np.max(self.diff_norm),' [mm]')
        print('Error min: ',np.min(self.diff_norm),' [mm]')
        print('Error standaed deviation: ',np.std(self.diff_norm))

    def plot_gaussian(self):
        mu = np.mean(self.diff_norm)
        sigma = np.std(self.diff_norm)
        x = np.arange(-40., 40., 0.01)
        y = (1/ np.sqrt(2 * np.pi * sigma)) * np.exp(-(x-mu) ** 2/ (2*sigma))
        plt.plot(x,y)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def meter2milli(self,meter):
        return meter*1000

    def plot_estimate_result(self):
        fig = plt.figure()
        self.ax = Axes3D(fig)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        estimate_pos = self.get_trans_position()
        for pos in estimate_pos:
            tmp = pos.tolist()
            self.ax.plot([tmp[0]], [tmp[1]], [tmp[2]], "o", color="#00aa00", ms=4, mew=0.5)
        for pos in self.motion_action_points.T.tolist():
            self.ax.plot([pos[0]], [pos[1]], [pos[2]], "o", color="#aa0000", ms=4, mew=0.5)
        self.ax.set_xlim([-2.0,2.0])
        self.ax.set_ylim([-2.0,2.0])
        self.ax.set_zlim([-2.0,2.0])
        self.ax.view_init(azim=90, elev=-90)
        plt.show()

    def detect_force_plate_slope(self):
        action_point = np.array([[0,0,0,1],[0.2,0.,0.,1],[0.,0.2,0.,1]])
        plane_pos = []
        for point in action_point:
            plane_pos.append(np.dot(self.mat,point))
        plane = get_plane_parameter(plane_pos)
        plane_comp = get_plane_parameter(np.array([[0.,0.,0.],[1.,0.,0.],[0.,0.,1.]]))
        plane_slope = get_two_plane_angle(plane, plane_comp)
        print('####################')
        print('Force plate slope: ', rad2deg(plane_slope), 'deg')
        print('####################')
