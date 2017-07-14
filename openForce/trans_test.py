
# coding: utf-8

from coordinate_transform import *
import numpy as np

vec = [1.,1.,1.]
v = np.array(vec)
v = v/np.linalg.norm(v)
mat = get_rodrigues_rotation(v, np.pi/5)
get_rotation_info(mat)
