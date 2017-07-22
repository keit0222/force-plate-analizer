
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

def get_rotation(axis_name,theta): # theta[rad]
    if axis_name == 'x' or axis_name == 'X':
        return np.array([[1., 0., 0.],
                         [0., np.cos(theta), -np.sin(theta)],
                         [0., np.sin(theta), np.cos(theta)]])
    elif axis_name == 'y' or axis_name == 'Y':
        return np.array([[np.cos(theta), 0., np.sin(theta)],
                         [0., 1., 0.],
                         [-np.sin(theta), 0., np.cos(theta)]])
    elif axis_name == 'z' or axis_name == 'Z':
        return np.array([[np.cos(theta), -np.sin(theta), 0.],
                         [np.sin(theta), np.cos(theta), 0.],
                         [0., 0., 1.]])
    else:
        return None

def get_euler_rotation(alfa,beta,gamma):
    return get_rotation_x(alfa)*get_rotation_y(beta)*get_rotation_z(gamma)

def get_roll_pitch_yaw_rotation(roll,pitch,yaw):
    return get_rotation_z(yaw)*get_rotation_y(pitch)*get_rotation_x(roll)

def get_center_of_rotation_vec(rotaion_matrix):
    # la,v = np.linalg.eig(rotaion_matrix)
    la,v = linalg.eig(rotaion_matrix)
    for i,la_val in enumerate(la):
        # print(la_val,v[:,i],type(la_val))
        la_val = np.real_if_close(la_val, tol=100*np.absolute(la_val)).tolist()
        if isinstance(la_val, float):
            return np.real_if_close(v[:,i], tol=100*np.absolute(la_val))

def get_rotation_angle(rotation_matrix):
    return np.arccos(((np.trace(rotation_matrix)-1)/2))

def show_rotation_info(rotation_matrix):
    vec = get_center_of_rotation_vec(rotation_matrix)
    theta = get_rotation_angle(rotation_matrix)
    print('########### Rotaion matrix information ##########')
    print('The center of rotation vector: ',vec)
    print('Rotation angle', rad2deg(theta), '[deg]')

def get_position_info(simultaneous_trans_matrix):
    return np.array([simultaneous_trans_matrix[0,3],simultaneous_trans_matrix[1,3],simultaneous_trans_matrix[2,3]])

def get_rotation_info(rotation_matrix):
    vec = get_center_of_rotation_vec(rotation_matrix)
    theta = get_rotation_angle(rotation_matrix)
    return [vec,rad2deg(theta)]

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
    return np.array([[a11,a12,a13],
                     [a21,a22,a23],
                     [a31,a32,a33]])

def get_rotation_matrix_from_quaternion(q):
    a11 = np.power(q[0],2) + np.power(q[1],2) - np.power(q[2],2) - np.power(q[3],2)
    a12 = 2 * (q[1]*q[2] - q[0]*q[3])
    a13 = 2 * (q[1]*q[3] + q[0]*q[2])
    a21 = 2 * (q[1]*q[2] + q[0]*q[3])
    a22 = np.power(q[0],2) - np.power(q[1],2) + np.power(q[2],2) - np.power(q[3],2)
    a23 = 2 * (q[2]*q[3] - q[0]*q[1])
    a31 = 2 * (q[1]*q[3] - q[0]*q[2])
    a32 = 2 * (q[2]*q[3] + q[0]*q[1])
    a33 = np.power(q[0],2) - np.power(q[1],2) - np.power(q[2],2) + np.power(q[3],2)
    return np.array([[a11,a12,a13],
                     [a21,a22,a23],
                     [a31,a32,a33]])

def get_quaternion(n, theta):
    n = np.array(n)
    n = n/np.linalg.norm(n)
    tmp = np.sin(theta/2.)*np.array(n)
    return np.array([np.cos(theta/2.),tmp[0],tmp[1],tmp[2]])

# rotation -> transfer
def get_simultaneous_matrix(R, q):
    return np.array([[R[0,0], R[0,1], R[0,2], q[0]],
                     [R[1,0], R[1,1], R[1,2], q[1]],
                     [R[2,0], R[2,1], R[2,2], q[2]],
                     [0.,0.,0.,1]])

# rotation -> transfer
def get_simultaneous_matrix_trans_rotate(R, q):
    tmp = np.dot(R,q)
    return np.array([[R[0,0], R[0,1], R[0,2], tmp[0]],
                     [R[1,0], R[1,1], R[1,2], tmp[1]],
                     [R[2,0], R[2,1], R[2,2], tmp[2]],
                     [0.,0.,0.,1]])

def get_simultaneous_trans_matrix(q):
    return np.array([[0., 0., 0., q[0]],
                     [0., 0., 0., q[1]],
                     [0., 0., 0., q[2]],
                     [0.,0.,0.,1]])

def deg2rad(degree):
    return degree/180*np.pi

def rad2deg(radian):
    return radian/np.pi*180
