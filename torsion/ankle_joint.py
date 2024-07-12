from skimage.measure import regionprops, label
import numpy as np
#from . import get_centroid, write_image, bresenhamline
from torsion import get_centroid, write_image, bresenhamline
from mask import get_contour_points, get_most_distal_layer_ankle, translate_image_coord_to_world_coord, get_convex_area
from vector import get_angle_between_vectors


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


def calc_pma(mask_t, mask_f, ankle_left, out_t=None):
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
    pma : plafond malleolus angle in degrees
    """

    """ # FIRST STEP: SPAN THE REFERENCE PLANE IN THE LAYER WITH THE LARGEST DIAMETER OF THE TIBIA

    # find index of the layer with the largest diameter of the tibia
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
    normal_vec = normal_vec / np.linalg.norm(normal_vec) """
    
    # FIRST STEP: SPAN THE REFERENCE PLANE BETWEEN TWO POINTS IN THE LATERAL THIRD OF THE TIBIA JOINT SURFACE AND ONE POINT IN THE MEDIAL THIRD OF THE TIBIAL JOINT SURFACE

    # find index of the layer with the largest diameter of the tibia
    layer_largest_diameter = get_layer_with_largest_diameter(mask_t)
    # get the contour points of the mask in this layer
    contour_pts = get_contour_points(mask_t[layer_largest_diameter])
    # combine y and x coordinates into a list of tuples
    contour_points = list(zip(contour_pts[0], contour_pts[1]))
    # sort the list of tuples by the x-coordinate (second element of the tuple)
    contour_pts_sorted = sorted(contour_points, key=lambda point: point[1])

    # find first two contour points in the lateral third of the tibia joint surface
    if ankle_left == False:
        p1_plane = np.array([layer_largest_diameter, contour_pts_sorted[len(contour_pts_sorted)-1][0], contour_pts_sorted[len(contour_pts_sorted)-1][1]])
        p2_plane = np.array([layer_largest_diameter, contour_pts_sorted[len(contour_pts_sorted)-int(len(contour_pts_sorted)/6)][0], contour_pts_sorted[len(contour_pts_sorted)-int(len(contour_pts_sorted)/6)][1]])
        #p1_plane = np.array([layer_largest_diameter, contour_pts[0][int(len(contour_pts[0])/3)], contour_pts[1][int(len(contour_pts[1])/3)]])
        #p2_plane = np.array([layer_largest_diameter, contour_pts[0][int(2*len(contour_pts[0])/3)], contour_pts[1][int(2*len(contour_pts[1])/3)]])
    else:
        p1_plane = np.array([layer_largest_diameter, contour_pts_sorted[0][0], contour_pts_sorted[0][1]])
        p2_plane = np.array([layer_largest_diameter, contour_pts_sorted[int(len(contour_pts_sorted)/6)][0], contour_pts_sorted[int(len(contour_pts_sorted)/6)][1]])
        #p1_plane = np.array([layer_largest_diameter, contour_pts[0][0], contour_pts[1][0]])
        #p2_plane = np.array([layer_largest_diameter, contour_pts[0][int(len(contour_pts[0])/3)], contour_pts[1][int(len(contour_pts[1])/3)]])
    
    # check which layer is the next distal layer
    if get_convex_area(mask_t[layer_largest_diameter-1]) < get_convex_area(mask_t[layer_largest_diameter+1]):
        next_distal_layer = layer_largest_diameter - 1
    else:
        next_distal_layer = layer_largest_diameter + 1

    # check in the next distal layer whether it can still be considered as tibia joint surface
    # assume that the mask needs to have at least 4/7 of the convex area of the mask with the largest diameter to still be considered as tibia joint surface
    if get_convex_area(mask_t[next_distal_layer]) > 4/7 * get_convex_area(mask_t[layer_largest_diameter]):
        # get the contour points of the mask in this layer
        contour_pts = get_contour_points(mask_t[next_distal_layer])
        # combine y and x coordinates into a list of tuples
        contour_points = list(zip(contour_pts[0], contour_pts[1]))
        # sort the list of tuples by the x-coordinate (second element of the tuple)
        contour_pts_sorted = sorted(contour_points, key=lambda point: point[1])

        # find the third point for the plane in the medial third of the tibia joint surface
        if ankle_left == False:
            p3_plane = np.array([next_distal_layer, contour_pts_sorted[int(len(contour_pts_sorted)/3)][0], contour_pts_sorted[int(len(contour_pts_sorted)/3)][1]])
            #p3_plane = np.array([next_distal_layer, contour_pts[0][0], contour_pts[1][0]])
        else:
            p3_plane = np.array([next_distal_layer, contour_pts_sorted[len(contour_pts_sorted)-int(len(contour_pts_sorted)/3)][0], contour_pts_sorted[len(contour_pts_sorted)-int(len(contour_pts_sorted)/3)][1]])
            #p3_plane = np.array([next_distal_layer, contour_pts[0][int(2*len(contour_pts[0])/3)], contour_pts[1][int(2*len(contour_pts[1])/3)]])
    # otherwise take the third reference point for the plane also from the layer with the largest diameter
    else:
        if ankle_left == False:
            p3_plane = np.array([layer_largest_diameter, contour_pts_sorted[int(len(contour_pts_sorted)/3)][0], contour_pts_sorted[int(len(contour_pts_sorted)/3)][1]])
            #p3_plane = np.array([layer_largest_diameter, contour_pts[0][0], contour_pts[1][0]])
        else:
            p3_plane = np.array([layer_largest_diameter, contour_pts_sorted[len(contour_pts_sorted)-int(len(contour_pts_sorted)/3)][0], contour_pts_sorted[len(contour_pts_sorted)-int(len(contour_pts_sorted)/3)][1]])
            #p3_plane = np.array([layer_largest_diameter, contour_pts[0][int(2*len(contour_pts[0])/3)], contour_pts[1][int(2*len(contour_pts[1])/3)]])

    # calculate the normal vector of the plane
    vec_1 = p1_plane - p2_plane
    vec_2 = p1_plane - p3_plane
    normal_vec = np.cross(vec_1, vec_2)
    # normalize the normal vector
    normal_vec = normal_vec / np.linalg.norm(normal_vec)


    # SECOND STEP: FIND THE REFERENCE LINE BETWEEN MOST DISTAL POINTS OF TIBIA AND FIBULA

    # find index of the layer with the most distal point of the tibia
    dist_layer_tibia = get_most_distal_layer_ankle(mask_t)
    # find index of the layer with the most distal point of the fibula
    dist_layer_fibula = get_most_distal_layer_ankle(mask_f)
    # calculate center of mass of tibia and fibula in the corresponding layers
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
    pma = 90 - get_angle_between_vectors(vec, normal_vec)

    if out_t is not None:
        write_image(mask, out_t)

    return mask, pma

def calc_mikulicz(center_fh, mask_hf, mask_k, mask_at, hip_reference, knee_reference, ankle_reference, length_femur, length_tibia, out_t=None):
    """
    calculates the required points and the reference line on hip and ankle joint level
    for the Mikulicz line

    Parameters
    ----------
    center_fh : array
        center of the femoral head as 3D coordinates with axis orientation z,y,x
    mask_hf : array
        mask of the femur segmentation on hip level as 3D array
    mask_k : array
        mask of the femur and tibia segmentation on knee level as 3D array
    mask_at : array
        mask of the tibia segmentation on ankle level as 3D array
    hip_reference : sitk.Image
        MR image of the hip
    knee_reference : sitk.Image
        MR image of the knee
    ankle_reference : sitk.Image
        MR image of the ankle
    length_femur :
        length of the femur
    length_tibia :
        length of the tibia
    out_t : str
        output path where the DICOM image file of the mask(tibia and fibula)
        with the required points and the reference lines should be saved

    Returns
    -------
    
    """

    # FIRST STEP: FIND THE REFERENCE LINE BETWEEN THE CENTER OF THE FEMORAL HEAD AND THE CENTER OF THE DISTAL TIBIA

    # find index of the layer with the largest diameter of the tibia
    layer_tibia = get_layer_with_largest_diameter(mask_at)
    # calculate center of mass of tibia in the corresponding layers
    com_tibia = get_centroid(mask_at[layer_tibia])

    # transform points from layer mask to 3D mask
    com_tibia = (layer_tibia, com_tibia[0], com_tibia[1])

    # transform the two centroids to world coordinates
    center_fh_world = translate_image_coord_to_world_coord(center_fh, hip_reference)
    com_tibia_world = translate_image_coord_to_world_coord(com_tibia, ankle_reference)

    # calculate the vector corresponding to the Mikulicz line based on the two centroids
    vec_mikulicz_line = np.array([com_tibia_world[0]-center_fh_world[0], com_tibia_world[1]-center_fh_world[1], com_tibia_world[2]-center_fh_world[2]])


    # SECOND STEP: FIND THE CENTER OF THE KNEE JOINT; CALCULATE ITS DISTANCE TO THE MIKULICZ LINE



    # determine the dimensions of the masks
    hf_shape = mask_hf.shape
    k_shape = mask_k.shape
    at_shape = mask_at.shape
    # convert the length of the femur and of the tibia to a number of slices with the same length
    # BE CAREFUL: THIS IS JUST AN APPROXIMATION SINCE THE VECTOR USED TO CALCULATE THE LENGTH OF THE FEMUR AND TIBIA WAS NOT PERPENDICULAR TO THE SURFACE
    spacing = knee_reference.GetSpacing()
    z_spacing = spacing[2]
    # NEED TO IMPLEMENT THE CORRECT CALCULATION OF THE GAP SIZE. THE CURRENT IMPLEMENTATION IS NOT CORRECT!
    # define the size of the gap between the hip and the knee stack (number of slices along the z-axis)
    gap_size_hk = int(length_femur / z_spacing)
    # define the size of the gap between the knee and the ankle stack (number of slices along the z-axis)
    gap_size_ka = int(length_tibia / z_spacing)
    # fill the gaps with zeros
    gap_shape_hk = (gap_size_hk, hf_shape[1], hf_shape[2])
    gap_shape_ka = (gap_size_ka, at_shape[1], at_shape[2])
    gap_hk = np.zeros(gap_shape_hk)
    gap_ka = np.zeros(gap_shape_ka)
    # concatenate the masks with the gap in between along the z-axis
    mask = np.concatenate((mask_at, gap_ka, mask_k, gap_hk, mask_hf), axis=0)

    # adjust the z-coordinates of the reference points to the gap
    center_fh = (center_fh[0] + gap_size_hk + k_shape[0] + gap_size_ka + at_shape[0], center_fh[1], center_fh[2])
    com_tibia = (com_tibia[0], com_tibia[1], com_tibia[2])

    # add reference line between most distal points of the fibula and tibia to the mask
    line = bresenhamline([center_fh], com_tibia, max_iter=-1)
    for k in range(len(line)):
        mask[int(line[k, 0]), int(line[k, 1]), int(line[k, 2])] = 3

    # mark centroids in the mask
    mask[center_fh] = 5
    mask[com_tibia] = 5

    if out_t is not None:
        write_image(mask, out_t)

    return mask
