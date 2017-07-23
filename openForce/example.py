import numpy as np
import force_analyzer as fa
import calibrate

def main():
    t = calibrate.Calibrate()
    # print(t.mat)
    # force_cls = fa.forceAnalyzer('ForcePlate\\','test.csv')
    # motion_cls = fa.motionAnalyzer('optitrack\\','test.c3d')
    # # force_cls.plot()
    # peek_time = force_cls.get_peek_time()
    # motion_cls.set_peek_time(peek_time)
    #
    # force_plate_action_points = np.array(force_cls.get_peek_action_point_for_trans()).T
    #
    # # force_cls.plot_peek_action_point()
    # motion_action_points = np.array(motion_cls.get_action_point_for_trans()).T
    #
    # print('Motion data',motion_action_points.shape,'inverse Force plate',np.linalg.pinv(force_plate_action_points).shape)
    #
    # # motion_action_points = M*force_plate_action_points
    # # M : 一次変換行列
    # # M = motion_action_points*pinv(force_plate_action_points)
    # M = np.dot(motion_action_points,np.linalg.pinv(force_plate_action_points))
    # print(M)

if __name__ == '__main__':
    main()
