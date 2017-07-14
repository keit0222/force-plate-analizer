
# coding: utf-8

import numpy as np
from scipy import linalg

def get_rotation_x(alfa): # alfa[rad]
    return np.array([[1., 0., 0.],
                     [0., np.cos(alfa), -np.sin(alfa)],
                     [0., np.sin(alfa), np.cos(alfa)]])

def get_rotation_y(beta): # beta[rad]
    return np.array([[np.cos(beta), 0., np.sin(beta)],
                     [0., 1., 0.],
                     [-np.sin(beta), 0., np.cos(beta)]])

def get_rotation_z(gamma): # gamma[rad]
    return np.array([[np.cos(gamma), -np.sin(gamma), 0.],
                     [np.sin(gamma), np.cos(gamma), 0.],
                     [0., 0., 1.]])

def get_euler_rotation(alfa,beta,gamma):
    return get_rotation_x(alfa)*get_rotation_y(beta)*get_rotation_z(gamma)

def get_roll_pitch_yaw_rotation(roll,pitch,yaw):
    return get_rotation_z(yaw)*get_rotation_y(pitch)*get_rotation_x(roll)

def get_center_of_rotation_vec(rotaion_matrix):
    # la,v = np.linalg.eig(rotaion_matrix)
    la,v = linalg.eig(rotaion_matrix)
    for i,la_val in enumerate(la):
        print(la_val,v[:,i],type(la_val))
        la_val = np.real_if_close(la_val, tol=100*np.absolute(la_val)).tolist()
        if isinstance(la_val, float):
            return np.real_if_close(v[:,i], tol=100*np.absolute(la_val))

def get_rotation_angle(rotation_matrix):
    return np.arccos(((np.trace(rotation_matrix)-1)/2))

def get_rotation_info(rotation_matrix):
    vec = get_center_of_rotation_vec(rotation_matrix)
    theta = get_rotation_angle(rotation_matrix)
    print('########### Rotaion matrix information ##########')
    print('The center of rotation vector: ',vec)
    print('Rotation angle', theta/np.pi*180, '[deg]')

def get_rodrigues_rotation(n, theta): # n: the center of rotation vector, theta[rad]
    n = np.array(n)
    n = n/np.linalg.norm(n)
    # print(n)
    a11 = np.cos(theta)+np.power(n[0],2)*(1-np.cos(theta))
    a12 = -n[2]*np.sin(theta) + n[0]*n[1]*(1-np.cos(theta))
    a13 = n[1]*np.sin(theta)+n[0]*n[2]*(1-np.cos(theta))
    a21 = n[2]*np.sin(theta)+n[0]*n[1]*(1-np.cos(theta))
    a22 = np.cos(theta)+np.power(n[1],2)*(1-np.cos(theta))
    a23 = -n[0]*np.sin(theta)+n[1]*n[2]*(1-np.cos(theta))
    a31 = -n[1]*np.sin(theta) + n[2]*n[0]*(1-np.cos(theta))
    a32 = n[0]*np.sin(theta)+n[1]*n[2]*(1-np.cos(theta))
    a33 = np.cos(theta)+np.power(n[2],2)*(1-np.cos(theta))
    mat = np.array([[a11,a12,a13],
                     [a21,a22,a23],
                     [a31,a32,a33]])
    return mat
