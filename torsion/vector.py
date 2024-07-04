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


def get_angle_between_vector_and_plane(vector, normal_vector):
    """ calculates the angle between a vector and a plane given the vector and the normal vector of the plane """
    # calculate the dot product of the vector and the normal vector
    dot_product = np.dot(vector, normal_vector)
    
    # calculate the magnitudes of the vector and the normal vector
    magnitude_vector = np.linalg.norm(vector)
    magnitude_normal = np.linalg.norm(normal_vector)
    
    # calculate the cosine of the angle
    cosine_angle = dot_product / (magnitude_vector * magnitude_normal)
    
    # use arccosine to find the angle in radians
    angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    # convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)

    # since we calculated the angle between the vector and the normal vector of the plane, we need to calculate the actual angle between the vector and the plane
    angle_vec_plane = 90 - angle_degrees
    
    return angle_vec_plane
