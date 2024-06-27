from skimage.measure import regionprops, label
import numpy as np
#from . import get_centroid, write_image, bresenhamline
from torsion import get_centroid, write_image, bresenhamline
from mask import get_contour_points, get_most_distal_layer
from vector import get_angle_between_vector_and_plane


def get_layer_with_largest_diameter(mask):
    """
    returns the layer with the largest diameter of a circle with the same area

    Parameters
    ----------
    mask : array
        mask as 3D array

    Returns
    -------
    int
        z coord of the layer
    """

    diameter = np.zeros(mask.shape[0])
    # save diameters of the layers
    for k in range(len(mask)):
        if len(np.nonzero(mask[k])[0]) != 0:
            props = regionprops(label(mask[k]))
            if props.__len__() > 1:
                i_biggest = 0
                for i in range(props.__len__()):
                    if props[i].equivalent_diameter > props[i_biggest].equivalent_diameter:
                        i_biggest = i
                diameter[k] = props[i_biggest].equivalent_diameter
            else:
                diameter[k] = props[0].equivalent_diameter

    # find index of the layer with the biggest diameter
    indices = np.argsort(diameter)
    return indices[-1]


def calc_ankle_joint(mask_t, mask_f, out_t=None):
    """
    calculates the required points and the reference line on ankle joint level
    for the measurement of the tibiatorsion

    Parameters
    ----------
    mask_t : array
        mask of the tibia segmentation on ankle level as 3D array
    mask_f : array
        mask of the fibula segmentation on ankle level as 3D array
    out_t : str
        output path where the DICOM image file of the mask(tibia and fibula)
        with the required points and the reference line should be saved

    Returns
    -------
    com_tibia : (int, int, int)
        centroid of pilon tibiale
    com_fibula : (int, int, int)
        centroid of fibula on selected layer
    """

    # find index of the layer with the biggest diameter of the tibia
    layer = get_layer_with_largest_diameter(mask_t)

    # calculate center of mass of tibia und fibula on the layer
    com_tibia = get_centroid(mask_t[layer])
    com_fibula = get_centroid(mask_f[layer])

    # mask with segmentation of tibia and fibula together
    mask = mask_t + mask_f

    # add reference line between centroids to the mask
    line = bresenhamline([com_tibia], com_fibula, max_iter=-1)
    for k in range(len(line)):
        mask[layer, int(line[k, 0]), int(line[k, 1])] = 3

    # transform points from layer mask to 3D mask
    com_tibia = (layer, com_tibia[0], com_tibia[1])
    com_fibula = (layer, com_fibula[0], com_fibula[1])

    # mark centroids in the mask
    mask[com_tibia] = 5
    mask[com_fibula] = 5

    if out_t is not None:
        write_image(mask, out_t)

    return mask, com_tibia, com_fibula


def calc_pma(mask_t, mask_f, out_t=None):
    """
    calculates the required points and the reference lines on ankle joint level
    for the measurement of the plafond malleolus angle

    Parameters
    ----------
    mask_t : array
        mask of the tibia segmentation on ankle level as 3D array
    mask_f : array
        mask of the fibula segmentation on ankle level as 3D array
    out_t : str
        output path where the DICOM image file of the mask(tibia and fibula)
        with the required points and the reference lines should be saved

    Returns
    -------
    pma: plafond malleolus angle in degrees
    """

    # FIRST STEP: SPAN THE REFERENCE PLANE IN THE LAYER WITH THE BIGGEST DIAMETER OF THE TIBIA

    # find index of the layer with the biggest diameter of the tibia
    layer_plane = get_layer_with_largest_diameter(mask_t)
    # get the contour points of the mask in this layer
    contour_pts = get_contour_points(mask_t[layer_plane])

    # create a plane with three points from the contour of the tibia
    p1_plane = np.array([layer_plane, contour_pts[0][0], contour_pts[1][0]])
    p2_plane = np.array([layer_plane, contour_pts[0][int(len(contour_pts[0])/3)], contour_pts[1][int(len(contour_pts[1])/3)]])
    p3_plane = np.array([layer_plane, contour_pts[0][int(2*len(contour_pts[0])/3)], contour_pts[1][int(2*len(contour_pts[1])/3)]])
    # calculate the normal vector of the plane
    vec_1 = p1_plane - p2_plane
    vec_2 = p1_plane - p3_plane
    normal_vec = np.cross(vec_1, vec_2)
    # normalize the normal vector
    normal_vec = normal_vec / np.linalg.norm(normal_vec)


    # SECOND STEP: FIND THE REFERENCE LINE BETWEEN MOST DISTAL POINTS OF TIBIA AND FIBULA

    # find index of the layer with the most distal point of the tibia
    dist_layer_tibia = get_most_distal_layer(mask_t)
    # find index of the layer with the most distal point of the fibula
    dist_layer_fibula = get_most_distal_layer(mask_f)
    # calculate center of mass of tibia und fibula on the corresponding layers
    com_tibia = get_centroid(mask_t[dist_layer_tibia])
    com_fibula = get_centroid(mask_f[dist_layer_fibula])

    # mask with segmentation of tibia and fibula together
    mask = mask_t + mask_f

    # transform points from layer mask to 3D mask
    com_tibia = (dist_layer_tibia, com_tibia[0], com_tibia[1])
    com_fibula = (dist_layer_fibula, com_fibula[0], com_fibula[1])

    # add reference line between most distal points of the fibula and tibia to the mask
    line = bresenhamline([com_tibia], com_fibula, max_iter=-1)
    for k in range(len(line)):
        mask[int(line[k, 0]), int(line[k, 1]), int(line[k, 2])] = 3
        #mask[layer, int(line[k, 0]), int(line[k, 1])] = 3

    # connect the points that were selected from the tibia contour to create the plane
    line_p1 = bresenhamline([p1_plane], p2_plane, max_iter=-1)
    for k in range(len(line_p1)):
        mask[int(line_p1[k, 0]), int(line_p1[k, 1]), int(line_p1[k, 2])] = 3

    line_p2 = bresenhamline([p2_plane], p3_plane, max_iter=-1)
    for k in range(len(line_p2)):
        mask[int(line_p2[k, 0]), int(line_p2[k, 1]), int(line_p2[k, 2])] = 3
    
    line_p3 = bresenhamline([p3_plane], p1_plane, max_iter=-1)
    for k in range(len(line_p3)):
        mask[int(line_p3[k, 0]), int(line_p3[k, 1]), int(line_p3[k, 2])] = 3

    # mark centroids in the mask
    mask[com_tibia] = 5
    mask[com_fibula] = 5
    mask[tuple(p1_plane)] = 5
    mask[tuple(p2_plane)] = 5
    mask[tuple(p3_plane)] = 5

    # calculate the Plafond Malleolus Angle
    vec = np.array([com_fibula[0]-com_tibia[0], com_fibula[1]-com_tibia[1], com_fibula[2]-com_tibia[2]])
    pma = get_angle_between_vector_and_plane(vec, normal_vec)

    if out_t is not None:
        write_image(mask, out_t)

    return mask, pma
