import SimpleITK as sitk
from .vector import round_to_int, get_vector, angle_between
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
