import math
import numpy as np


def length(v):
    """ returns the length of the vector 'v' """
    return math.sqrt(np.vdot(v, v))


def angle_between(v1, v2):
    """ returns the angle in radians between vectors 'v1' and 'v2' """
    return math.acos(np.vdot(v1, v2) / (length(v1) * length(v2)))


def get_vector(pt1, pt2):
    """ returns 2D vector between two points """
    return np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]])


def round_to_int(pt):
    """ round a 2D or 3D point with float coordinates to int """
    if len(pt) == 2:
        return np.array([int(round(pt[0])), int(round(pt[1]))])
    if len(pt) == 3:
        return np.array([int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))])
