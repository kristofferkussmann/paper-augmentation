import SimpleITK as sitk
#from .vector import round_to_int, get_vector, angle_between
from vector import round_to_int, get_vector, angle_between
from scipy.ndimage.morphology import binary_erosion
from skimage.measure import regionprops, label
from skimage.transform import rotate
import numpy as np
import math


def get_mask(path):
    """ returns the mask of a DICOM image file as an array """
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image)


def write_image(mask, path, spacing=None):
    """
    saves mask as DICOM image file

    Parameters
    ----------
    mask : array
    path : str
        output path where the DICOM image file should be saved

    """
    img = sitk.GetImageFromArray(mask)
    if spacing is not None:
        img.SetSpacing(spacing)
    sitk.WriteImage(img, path)


def get_contour(mask):
    """ returns the contour of a mask """
    eroded_mask = binary_erosion(mask)
    diff_mask = mask - eroded_mask
    return diff_mask


def get_contour_points(mask, output_path=None):
    """ returns the contour coordinates of a mask """
    diff_mask = get_contour(mask)
    if output_path is not None:
        write_image(diff_mask, output_path)
    return np.nonzero(diff_mask)


def get_side_contour_pts(mask, y: int):
    """
    returns the smallest and largest x coordinate of the contour
    for a given y in a 2D mask
    """
    nonzero = np.nonzero(mask[y])[0]
    return nonzero.min(), nonzero.max()


def get_dorsal_mask_pt(mask):
    """
    returns point of the 2D mask with the highest y value -> most dorsal point
    """
    pts = get_contour_points(mask)
    indices = np.argsort(pts[0])

    return pts[0][indices[-1]], pts[1][indices[-1]]


def get_centroid(mask):
    """ calculates the centroid of the 2d mask """
    props = regionprops(label(mask))
    if props.__len__() > 1:
        i_biggest = 0
        for i in range(props.__len__()):
            if props[i].area > props[i_biggest].area:
                i_biggest = i
        com = props[i_biggest].centroid
    else:
        com = props[0].centroid
    com = round_to_int(com)
    return com


def get_layer_with_biggest_convex_area(mask):
    """ returns z coordinate of the layer with the biggest convex area """
    area = np.zeros(mask.shape[0])
    # save diameters of the layers
    for k in range(len(mask)):
        if len(np.nonzero(mask[k])[0]) != 0:
            props = regionprops(label(mask[k]))
            if props.__len__() > 1:
                i_biggest = 0
                for i in range(props.__len__()):
                    if props[i].convex_area > props[i_biggest].convex_area:
                        i_biggest = i
                area[k] = props[i_biggest].convex_area
            else:
                area[k] = props[0].convex_area

    # find index of the layer with the biggest diameter
    indices = np.argsort(area)
    return indices[-1]


def rotate_pt(origin, point, angle, deg=True):
    """
    Rotate a point on a layer(2D) counterclockwise by a given angle
    around a given origin
    """

    if deg:
        angle = np.deg2rad(angle)
    oy, ox = origin
    py, px = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (-py + oy)
    qy = oy - math.sin(angle) * (px - ox) - math.cos(angle) * (-py + oy)
    return qy, qx


def transform_pt(pt, origin, angle, offset=None):
    """
    Transform a point on a layer(2D) into the rotated mask
    """
    new_pt = rotate_pt(origin, pt, angle)

    if offset is not None:
        new_pt = np.array(new_pt) + np.array(offset)

    return new_pt


def rotate_mask_vec_parallel(mask, vec1, vec2, return_angle=True):
    """
    rotates the 2D mask so that the given vectors are parallel
    and returns the rotated mask and optionally the rotation angle
    """

    angle = np.rad2deg(angle_between(vec1, vec2))

    # need to rotate clockwise or counterclockwise?
    if np.rad2deg(angle_between(rotate_pt((0, 0), vec1, angle), vec2)) != 0:
        angle = -angle

    rotated_mask = rotate(mask, angle, resize=True, preserve_range=True) > 0
    rotated_mask = rotated_mask.astype(np.uint8)

    if return_angle:
        return rotated_mask, angle
    else:
        return rotated_mask


def rotate_mask_dorsal_pts(mask, thresh_pt):
    """
    rotate the mask so that a line between the most dorsal point right and left
    (on x axis)from the notch would be parallel to the x axis
    """
    start = get_dorsal_mask_pt(mask)
    if start[1] < thresh_pt[1]:
        end = get_dorsal_mask_pt(mask[:, thresh_pt[1]+3:])
        end = (end[0], end[1] + thresh_pt[1])
    else:
        end = start
        start = get_dorsal_mask_pt(mask[:, :thresh_pt[1]-3])

    return rotate_mask_vec_parallel(mask, get_vector(start, end), np.array([0, 1]))


def calc_discontinuity(mask, y: int):
    """
    finds discontinuity in the mask for a given y coordinate

    Returns
    -------
    int
        x coordinates of the discontinuity
    """
    side_pts = get_side_contour_pts(mask, y)
    return np.nonzero(1-mask[y, side_pts[0]:side_pts[1]])[0] + side_pts[0]


def determine_min_y(mask, percentage=0.5):
    """
    calculate a min y value to just have a look at the dorsal part of the mask

    Parameters
    ----------
    mask : array
    percentage : int
        percentage of the dorsal y values
    """
    contour_pts = get_contour_points(mask)

    num_pts = int(len(contour_pts[0]) * percentage)

    sorted_y = np.sort(contour_pts[0])
    return sorted_y[-num_pts]


def find_notch(mask, min_y=None, percentage=0.5, thresh=0, break_after_first=False):
    """
    searches a notch by decreasing the y value (max to the value of min_y)
    -> moving to ventral to find the notch
    """
    # min_y is the threshold for the most ventral point where the notch can be
    if min_y is None:
        min_y = determine_min_y(mask, percentage)
    y, _ = get_dorsal_mask_pt(mask)
    y = int(y)

    return_allowed = False
    notch = None, None

    # lowest mask pt >= y coord of notch >= min_y
    while y >= min_y:
        discontinuity = calc_discontinuity(mask, y)

        # discontinuity found, return is now possible
        if isinstance(discontinuity, np.ndarray) and len(discontinuity) > thresh:
            return_allowed = True

        # no more discontinuity was found -> calculate discontinuity of
        # previous level and return its center
        else:
            if return_allowed:
                new_discontinuity = calc_discontinuity(mask, y+1)
                notch = (y+1, new_discontinuity[int(len(new_discontinuity)/2)])
                return_allowed = False
                if break_after_first:
                    break
        y -= 1
    return notch


def sep_seg(mask, value):
    """
    Divides the mask in the middle and returns both separately with only 1's
    where the original mask has 'value'
    """
    mask_left = mask.copy()
    mask_left = (mask_left == value).astype(np.uint16)
    mask_left[:, :, int((mask.shape[2] - 1) / 2):] = 0

    mask_right = mask.copy()
    mask_right = (mask_right == value).astype(np.uint16)
    mask_right[:, :, :int((mask.shape[2] - 1) / 2)] = 0

    return mask_left, mask_right


def get_diameter(mask):
    """
    returns the diameter of the given 2D mask
    """
    props = regionprops(label(mask))
    diameter = props[0].equivalent_diameter
    return diameter


def get_most_distal_layer_ankle(mask):
    """
    returns the layer with the most distal point of the ankle mask

    Parameters
    ----------
    mask : array
        mask as 3D array

    Returns
    -------
    int
        z coord of the most distal layer
    """

    # z-coordinate of the layer with the most distal point of the mask
    z_coord_layer = 0

    # check whether slice with index 0 or max_index is the most distal slice of the MR image
    # assume that in the most distal slice of the ankle measurement there is no mask but in the most proximal slice there is
    if mask[len(mask) - 1].max() == 1:
        # max_index corresponds to most proximal layer
        # start iterating from index 0
        for k in range(len(mask)):
            if mask[k].max() == 1:
                z_coord_layer = k
                break
    else:
        # max_index corresponds to most distal layer
        # start iterating from max_index
        for k in range(len(mask) - 1, -1, -1):
            if mask[k].max() == 1:
                z_coord_layer = k
                break

    return z_coord_layer


def get_most_distal_layer_hip(mask):
    """
    returns the layer with the most distal point of the hip mask

    Parameters
    ----------
    mask : array
        mask as 3D array

    Returns
    -------
    int
        z coord of the most distal layer
    """

    # z-coordinate of the layer with the most distal point of the mask
    z_coord_layer = 0

    # check whether slice with index 0 or max_index is the most distal slice of the MR image
    # assume that in the most proximal slice of the hip measurement there is no mask but in the most distal slice there is
    if mask[len(mask) - 1].max() == 1:
        # max_index corresponds to most distal layer
        z_coord_layer = len(mask) - 1
    else:
        # max_index corresponds to most proximal layer
        z_coord_layer = 0

    return z_coord_layer


def get_most_proximal_layer_knee_femur(mask):
    """
    returns the layer with the most proximal point of the knee femur mask

    Parameters
    ----------
    mask : array
        mask as 3D array

    Returns
    -------
    int
        z coord of the most proximal layer
    """

    # z-coordinate of the layer with the most proximal point of the mask
    z_coord_layer = 0

    # assume that the diameter of the knee femur in the more distal slices is larger than in the proximal slices

    # get first candidate for most proximal slice
    for k in range(len(mask)):
        if mask[k].max() == 1:
            z_candidate_1 = k
            # around the knee joint, there might be a slice where there still is a small femur mask
            # increase z_coordinate by 2 to get a larger femur mask for the comparison of the two candidates
            # otherwise it might happen that the size of the most proximal and most distal mask does not differ significantly
            z_candidate_1_1 = k + 2
            break
    
    # get second candidate for most proximal slice
    for k in range(len(mask) - 1, -1, -1):
        if mask[k].max() == 1:
            z_candidate_2 = k
            # see motivation when getting first candidate
            z_candidate_2_1 = k - 2
            break
    
    diameter_1 = get_diameter(mask[z_candidate_1_1])
    diameter_2 = get_diameter(mask[z_candidate_2_1])

    if diameter_1 < diameter_2:
        z_coord_layer = z_candidate_1
    else:
        z_coord_layer = z_candidate_2

    return z_coord_layer


def get_length(hip_reference: sitk.Image, knee_reference: sitk.Image, ankle_reference: sitk.Image, hip_seg: sitk.Image, knee_seg: sitk.Image, ankle_seg: sitk.Image):
    """
    calculates the length of the femur and tibia of both legs

    Parameters
    ----------
    hip_reference: sitk.Image
        image of the hip region
    knee_reference:sitk.Image
        image of the knee region
    ankle_reference:sitk.Image
        image of the ankle region
    hip_seg: sitk.Image
        segmentation of the hip region
    knee_seg:sitk.Image
        segmentation of the knee region
    ankle_seg:sitk.Image
        segmentation of the ankle region

    Returns
    -------
    left_femur: length of the left femur
    right_femur: length of the right femur
    left_tibia: length of the left tibia
    right_tibia: length of the right tibia
    left: length left femur + length left tibia
    right: length right femur + length right tibia
    """

    #logger = logging.getLogger('app')

    hip_array = sitk.GetArrayFromImage(hip_seg)
    knee_array = sitk.GetArrayFromImage(knee_seg)
    ankle_array = sitk.GetArrayFromImage(ankle_seg)

    # HIP

    left_arr_hip = hip_array[:, :, :int(hip_array.shape[2] / 2)]
    right_arr_hip = hip_array[:, :, int(hip_array.shape[2] / 2):]

    left_proximal_femur_point = np.argwhere(left_arr_hip == 1)[-1] # probably better to find center point
    right_proximal_femur_point = np.argwhere(right_arr_hip == 1)[-1]
    
    # KNEE
    # assume label number of femur is 1 and label number of tibia is 2

    left_arr_knee = knee_array[:, :, :int(knee_array.shape[2] / 2)]
    right_arr_knee = knee_array[:, :, int(knee_array.shape[2] / 2):]

    left_distal_femur_point = np.argwhere(left_arr_knee == 1)[0]
    right_distal_femur_point = np.argwhere(right_arr_knee == 1)[0]

    left_proximal_tibia_point = np.argwhere(left_arr_knee == 2)[-1]
    right_proximal_tibia_point = np.argwhere(right_arr_knee == 2)[-1]

    # ANKLE
    # assume label number of tibia is 1 and label number of fibula is 2

    left_arr_ankle = ankle_array[:, :, :int(ankle_array.shape[2] / 2)]
    right_arr_ankle = ankle_array[:, :, int(ankle_array.shape[2] / 2):]

    left_distal_tibia_point = np.argwhere(left_arr_ankle == 1)[0]
    right_distal_tibia_point = np.argwhere(right_arr_ankle == 1)[0]

    # CALCULATE LENGTH

    left_femur_length = np.linalg.norm(translate_image_coord_to_world_coord(left_proximal_femur_point, hip_reference) - 
                            translate_image_coord_to_world_coord(left_distal_femur_point, knee_reference))

    right_femur_length = np.linalg.norm(translate_image_coord_to_world_coord(right_proximal_femur_point, hip_reference) - 
                            translate_image_coord_to_world_coord(right_distal_femur_point, knee_reference))     

    left_tibia_length = np.linalg.norm(translate_image_coord_to_world_coord(left_proximal_tibia_point, knee_reference) - 
                            translate_image_coord_to_world_coord(left_distal_tibia_point, ankle_reference))

    right_tibia_length = np.linalg.norm(translate_image_coord_to_world_coord(right_proximal_tibia_point, knee_reference) - 
                            translate_image_coord_to_world_coord(right_distal_tibia_point, ankle_reference))

    return {'left_femur': left_femur_length, 'right_femur': right_femur_length, 'left_tibia': left_tibia_length, 
            'right_tibia': right_tibia_length, 'left': left_femur_length + left_tibia_length, 'right': right_femur_length +
            right_tibia_length}


def translate_image_coord_to_world_coord(image_coord: np.array, reference_image: sitk.Image) -> np.array:
    """
    translates image to world coordinates
    """
    #logger = logging.getLogger('app')
    image_coord = np.array(list(reversed(image_coord)))
    D_ = reference_image.GetDirection()
    D = np.empty((3,3))
    D[0] = np.array(D_[0:3])
    D[1] = np.array(D_[3:6])
    D[2] = np.array(D_[6:])
    S = reference_image.GetSpacing()
    O = reference_image.GetOrigin()
    S2 = np.zeros((3,3))
    np.fill_diagonal(S2, S)

    world_coordinates = D @ S2 @ image_coord.T + O
    return world_coordinates
