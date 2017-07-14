
# coding: utf-8

from coordinate_transform import *
import numpy as np

vec = [-4.,-3.,-1.]
mat = get_rotation_matrix_from_quaternion(get_quaternion(vec, np.pi/5))
get_rotation_info(mat)
