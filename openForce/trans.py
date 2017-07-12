
# coding : utf-8

import force_analyzer as fa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Trans():
    def __init__(self):
        self.force_cls = fa.forceAnalyzer('ForcePlate\\','test.csv')
        self.motion_cls = fa.motionAnalyzer('optitrack\\','test.c3d')
        self.diff_norm = []
        peek_time = self.force_cls.get_peek_time()
        self.motion_cls.set_peek_time(peek_time)
        self.force_plate_action_points = 0
        self.motion_action_points = 0
        self.mat = self.get_trans()
        self.trans_check()
        self.evaluate_trans()
        self.plot_gaussian()

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
