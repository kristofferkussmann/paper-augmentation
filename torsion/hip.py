from . import write_image, get_centroid
from .mask import get_contour_points, get_contour, find_notch, \
    rotate_mask_dorsal_pts, transform_pt
from .vector import round_to_int, length, get_vector
import numpy as np
import math
from scipy.optimize import curve_fit
from . import bresenhamline
from skimage import measure
from skimage.transform import rotate
import pandas as pd
from .knee import calc_knee
import cv2
from scipy import optimize


def sphere_fit(sp_x, sp_y, sp_z):
    """
    fit a sphere to X,Y, and Z data points
    returns the radius and center points of
    the best fit sphere
    """
    #   Assemble the A matrix
    sp_x = np.array(sp_x)
    sp_y = np.array(sp_y)
    sp_z = np.array(sp_z)
    A = np.zeros((len(sp_x), 4))
    A[:, 0] = sp_x*2
    A[:, 1] = sp_y*2
    A[:, 2] = sp_z*2
    A[:, 3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(sp_x), 1))
    f[:, 0] = (sp_x*sp_x) + (sp_y*sp_y) + (sp_z*sp_z)
    C, residules, rank, singval = np.linalg.lstsq(A, f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2]


def circle_fit(mask: np.array):
    """Fit a circle to an arbitrary 2D point cloud.

    Args:
        mask (np.array): A 2D numpy array with axes orientation y,x. E.g. the outline of a segmentation mask.

    Returns:
        tuple(np.array, float): A tuple containing the center of the fitted circle (orientation y,x) and the radius of the circle.
    """
    #point_cloud = np.argwhere(mask != 0)
    y = mask[:,0]
    x = mask[:,1]

    y_m = np.mean(y)
    x_m = np.mean(x)

    def calc_r(xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)
    
    def f_2(c):
        ri = calc_r(*c)
        return ri - ri.mean()
    
    center_estimate = y_m, x_m
    center, _ = optimize.leastsq(f_2, center_estimate)
    radius = calc_r(*center).mean()
    return np.flip(center), radius


def draw_sphere(mask, r, center, z_ratio):
    """ marks the points of a sphere with a given radius and center in the mask """
    for z in range(mask.shape[0]):  # all z coordinates
        for x in range(center[2]-int(r)-2, center[2]+int(r)+2):  # all relevant x coordinates
            if r**2 - (x-center[2])**2 - (z_ratio*(z-center[0]))**2 >= 0:
                y = int(round(math.sqrt(
                    r**2 - (x-center[2])**2 - (z_ratio*(z-center[0]))**2)))
                if y < mask.shape[1]:
                    mask[z, center[1]+y, x] = 5
                    mask[z, center[1]-y, x] = 5

        for y in range(center[1]-int(r)-2, center[1]+int(r)+2):  # all relevant y coordinates
            if r**2 - (y-center[1])**2 - (z_ratio*(z-center[0]))**2 >= 0:
                x = int(round(math.sqrt(
                    r**2 - (y-center[1])**2 - (z_ratio*(z-center[0]))**2)))
                if x < mask.shape[2]:
                    mask[z, y, center[2]+x] = 5
                    mask[z, y, center[2]-x] = 5


def draw_circle(mask, layer, r, center):
    for x in range(center[1]-int(r)-2, center[1]+int(r)+2):  # all relevant x coordinates
        if r ** 2 - (x - center[1]) ** 2 >= 0:
            y = int(round(math.sqrt(r ** 2 - (x - center[1]) ** 2)))
            if y < mask.shape[1]:
                mask[layer, center[0] + y, x] = 5
                mask[layer, center[0] - y, x] = 5

    for y in range(center[0]-int(r)-2, center[0]+int(r)+2):  # all relevant y coordinates
        if r ** 2 - (y - center[0]) ** 2 >= 0:
            x = int(round(math.sqrt(r ** 2 - (y - center[0]) ** 2)))
            if x < mask.shape[2]:
                mask[layer, y, center[1] + x] = 5
                mask[layer, y, center[1] - x] = 5


def pts_on_circle(mask, r, center):
    """

    Parameters
    ----------
    mask
        2d mask
    r
        radius
    center

    Returns
    -------

    """
    rt = False
    for x in range(center[1]-int(r)-2, center[1]+int(r)+2):  # all relevant x coordinates
        temp = r**2 - (x-center[1])**2
        if temp > 0 and (mask[int(round(center[0] - math.sqrt(temp))), x] != 0 or
                         mask[int(round(center[0] + math.sqrt(temp))), x] != 0):
            rt = True
            break
    if rt:
        return rt
    else:
        for y in range(center[0]-int(r)-2, center[0]+int(r)+2):  # all relevant y coordinates
            temp = r**2 - (y-center[0]) ** 2
            if temp > 0 and (mask[y, int(round(center[1] - math.sqrt(temp)))] != 0 or
                             mask[y, int(round(center[1] + math.sqrt(temp)))] != 0):
                rt = True
                break
    return rt


def contour_femoral_neck(mask, contour, layer_selected, center, r):
    """

    Parameters
    ----------
    mask
        3d segmentation mask
    contour
        contour of segmentation mask
    layer_selected
        index of layer with femoral neck
    center
        center coordinates of femur head
    r
        radius of femur head

    Returns
    -------

    """
    # rotate mask and find notch to regulate the radii of the spheres
    mask_new = mask[layer_selected].copy()
    rotated_mask, angle1 = rotate_mask_dorsal_pts(mask_new, get_centroid(mask_new))
    rotated_mask, angle2 = rotate_mask_dorsal_pts(rotated_mask, get_centroid(rotated_mask))
    angle = angle1 + angle2
    notch_rot = find_notch(rotated_mask, percentage=0.8, thresh=5, break_after_first=True)
    rot_offset = np.array([_rot_dim - _orig_dim
                           for _rot_dim, _orig_dim in zip(rotated_mask.shape,
                                                          mask_new.shape)])
    rot_center = (int((rotated_mask.shape[0] - 1) / 2),
                  int((rotated_mask.shape[1] - 1) / 2))
    if angle == 0:
        notch = notch_rot
    else:
        notch = transform_pt(notch_rot, rot_center, -angle,
                             offset=-rot_offset / 2)

    # get 2 new radii a little closer and further to the center than the notch
    a = length(get_vector(notch, [center[1], center[2]]))/r
    r_1 = (-0.1+a)*r
    r_2 = (0.1+a)*r

    mask_new = mask[layer_selected].copy()

    # get contour only between the two circles on the selected layer
    for w in range(mask.shape[2]-1):  # all x coordinates
        if r_2**2 - (w-center[2])**2 >= 0:
            y_b = int(round(math.sqrt(r_2**2 - (w-center[2])**2)))
            if center[1]+y_b < mask.shape[1]:
                contour[layer_selected, center[1]+y_b:, w] = 0
                mask_new[center[1]+y_b:, w] = 0
            if center[1]-y_b > 0:
                contour[layer_selected, :center[1]-y_b, w] = 0
                mask_new[:center[1]-y_b, w] = 0
        else:
            contour[layer_selected, :, w] = 0
            mask_new[:, w] = 0

        if r_1**2 - (w-center[2])**2 >= 0:
            y_c = int(round(math.sqrt(r_1**2 - (w-center[2])**2)))
            if center[1]-y_c > 0 and center[1]+y_c < mask.shape[1]:
                contour[layer_selected, center[1]-y_c:center[1]+y_c, w] = 0
                mask_new[center[1]-y_c:center[1]+y_c, w] = 0

    return mask_new, contour, r_1, r_2


def angle_between(v1, v2):
    """ returns the angle in radians between vectors 'v1' and 'v2' """
    return math.acos(np.vdot(v1, v2) / (length(v1) * length(v2)))


def find_femoral_neck_center(mask: np.array, left: bool, layer_index: int = 0) -> np.array:
    """Find the center point of the femoral neck.

    Args:
        mask (np.array): A 3D segmentation mask with axes orientation z,y,x.

    Returns:
        np.array: A 3D numpy array representing the center of the femoral neck with axes orientation z,y,x.
    """
    if layer_index == 0:
        for i in range(mask.shape[0]): # go proximal until a segmentation with two components is found, i.e. there is a gap
            if len(tmp := measure.find_contours(mask[i], 0.1)) >= 2:
                tmp = sorted(tmp, key=len, reverse=True)
                if (len(tmp[0]) < 25) or (len(tmp[1]) < 25):
                    continue

                # layer_index = i - 1
                layer_index = i - 2
                break

    # get the vectors of the top left and the top right point of the mask layer
    a = np.argwhere(mask[layer_index] != 0)
    tmp = a[a[:, 1].argsort()]
    left_vec = tmp[0]
    tmp = a[a[:, 0].argsort()]
    right_vec = tmp[0]

    tilt_axis = left_vec - right_vec  # get the vector tangent that runs along the upper outline of the mask
    compare_vec = np.array([0, 0]) - np.array([0, 1])  # get a vector that runs parallel to the x axis
    to_rotate = math.degrees(angle_between(tilt_axis, compare_vec))  # rotate the mask so that the tangent of the mask is parallel to the x axis

    rotated_mask = np.copy(mask[layer_index])
    rotated_mask = rotate(rotated_mask, -to_rotate if left else to_rotate)#.astype(int)
    # for i in range(len(rotated_mask)):
    #    rotated_mask[i] = rotate(rotated_mask[i], -to_rotate)
    
    """
    # find the y level where the gap between the femoral head and the femoral neck starts, and get the x values
    for i in range(max(np.argwhere(rotated_mask != 0)[:,0]), min(np.argwhere(rotated_mask != 0)[:,0]), -1):
        disc = rotation_utils.calc_discontinuity(rotated_mask, i)
        if len(disc) > 20:
            break

    x_index = np.median(np.array(disc)) # find the middle of the gap
    df = pd.DataFrame(data=np.argwhere(rotated_mask != 0), columns=['y', 'x'])
    rotated_mask[int(df.loc[df['x'] == int(x_index)]['y'].mean()), int(x_index)] = 1000
    """
    mask_coords = np.argwhere(rotated_mask != 0)
    df = pd.DataFrame({'x': [x[1] for x in mask_coords], 'y': [x[0] for x in mask_coords]})
    xrange = df['x'].max() - df['x'].min()
    ndf = df.loc[df['x'] > df['x'].min() + .30 * xrange].loc[df['x'] < df['x'].min() + .70 * xrange]  # only include mid 50%
    idx = (ndf.groupby('x').max() - ndf.groupby('x').min()).idxmin()  # find x coord where mask is thinnest
    y = ndf.groupby('x').mean().loc[idx]

    rotated_mask[int(y.to_numpy()[0]), int(idx.to_numpy()[0])] = 1000
    # femoral_neck_center = np.array([df.loc[df['x'] == int(x_index)]['y'].mean(), x_index]) # get the mean y value for the middle of the gap

    # theta = -np.deg2rad(to_rotate)
    # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # femoral_neck_center = np.dot(rotation_matrix, femoral_neck_center)
    rotated_mask = rotate(rotated_mask, to_rotate if left else -to_rotate)#.astype(int)
    femoral_neck_center = np.argwhere(rotated_mask == max(np.unique(rotated_mask)))[0]
    femoral_neck_center = np.array([layer_index, femoral_neck_center[0], femoral_neck_center[1]])
    return femoral_neck_center


def find_femoral_neck_center_reikeras(mask: np.array, femoral_head_idx: int) -> (np.array, int, tuple):
    """Find the center point of the femoral neck.

    Args:
        mask (np.array): A 3D segmentation mask with axes orientation z,y,x.
        femoral_head_idx (int): index of the layer where the femoral head center is located.
    Returns:
        np.array: A 3D numpy array representing the center of the femoral neck with axes orientation z,y,x.
    """
    last_angle = 1000
    candidate_slices = dict()
    candidate_centers = list()
    candidate_reference_points = list()

    for i in range(femoral_head_idx - 1, -1, -1):
        contours = measure.find_contours(mask[i])
        if len(contours) != 1:  # if there is more than one contour, skip this slice
            continue

        outline = contours[0]
        outline_df = pd.DataFrame({'x': [x[1] for x in outline], 'y': [x[0] for x in outline]})

        # split outline into lower and upper part
        lower_outline = np.fliplr(outline_df.groupby(by='x').max().reset_index().to_numpy())
        upper_outline = np.fliplr(outline_df.groupby(by='x').min().reset_index().to_numpy())

        # fit a line to each part
        lower_fit = np.polynomial.polynomial.Polynomial.fit(x=[x[1] for x in lower_outline],
                                                            y=[x[0] for x in lower_outline], deg=1)
        upper_fit = np.polynomial.polynomial.Polynomial.fit(x=[x[1] for x in upper_outline],
                                                            y=[x[0] for x in upper_outline], deg=1)

        # calculate the angle between the two lines
        start_point_lower = np.array([lower_fit(outline_df['x'].min()), outline_df['x'].min()])
        end_point_lower = np.array([lower_fit(outline_df['x'].max()), outline_df['x'].max()])
        start_to_end_lower = (end_point_lower - start_point_lower) * 0.5

        start_point_upper = np.array([upper_fit(outline_df['x'].min()), outline_df['x'].min()])
        end_point_upper = np.array([upper_fit(outline_df['x'].max()), outline_df['x'].max()])
        start_to_end_upper = (end_point_upper - start_point_upper) * 0.5

        halfway_point_lower = start_point_lower + start_to_end_lower  # get the halfway point of the lower line
        halfway_point_upper = start_point_upper + start_to_end_upper  # get the halfway point of the upper line

        axis_between = halfway_point_upper + (halfway_point_lower - halfway_point_upper)
        axis_distance = np.linalg.norm(axis_between)  # get the distance between the two cortical axes

        femoral_neck_center = halfway_point_upper + (halfway_point_lower - halfway_point_upper) * 0.5  # get the halfway point between the two lines

        angle = math.degrees(angle_between(start_to_end_lower, start_to_end_upper))
        if (angle < 5):# and (last_angle < 5): # if the angle of the previous slice and the current slice is less than 5 degrees, return the current slice
            candidate_slices[i] = axis_distance
            candidate_centers.append(femoral_neck_center)
            candidate_reference_points.append((start_point_lower, start_point_upper, end_point_lower, end_point_upper))

        last_angle = angle

    distances = np.array(list(candidate_slices.values()))
    max_distance_idx = np.argmax(distances)

    i = list(candidate_slices.keys())[max_distance_idx]
    femoral_neck_center = candidate_centers[max_distance_idx]
    reference_points = candidate_reference_points[max_distance_idx]

    return femoral_neck_center.astype(np.int64), i, reference_points


def calc_hip(m, z_ratio, femur_left, knee_mask=None, path_out=None, path_out_ball=None, mark_points: bool = True, reikeras=False):
    mask = np.copy(m)

    for i in range(mask.shape[0]):
        tmp = np.argwhere(mask > 0)
        for j in range(min(tmp[:, 1]), max(tmp[:, 1])):
            for k in range(min(tmp[:, 2]), max(tmp[:, 2])):
                nonzero_neighbours = mask[i, j-1, k] + mask[i, j+1, k] + mask[i, j, k-1] + mask[i, j, k+1]
                if nonzero_neighbours > 2:
                    # fill in pixels that (probably) should be segmented
                    mask[i, j, k] = 1

        # filter out small components
        connected_components = cv2.connectedComponentsWithStats(mask[i].astype(np.uint8), connectivity=4)
        label_map = connected_components[1]
        component_sizes = connected_components[2][:, -1]
        large_components = np.argwhere(component_sizes > 50)
        mask_copy = np.zeros(mask[i].shape)
        for component in large_components:
            mask_copy += np.where(label_map == component[0], component[0], 0)
        
        mask_copy = np.where(mask_copy > 0, 1, 0)
        mask[i] = mask_copy

    # Calculate mask of femur
    # mask = get_mask(path_in)
    contour_pts = get_contour_points(mask)

    # get highest layer with a mask point, its centroid
    # and lowest layer with mask point on this centroid
    layer_high = np.amax(contour_pts[0])
    # this is crap
    """
    while (len(np.argwhere(mask[layer_high - 1] == 1))) == 0 or (len(np.argwhere(mask[layer_high] == 1))):
        layer_high -= 1
    """

    com_high = get_centroid(mask[layer_high - 1])
    layer_low = layer_high
    while mask[layer_low - 1, com_high[0], com_high[1]] != 0:
        layer_low -= 1

    # get contour of each layer of the mask separately
    contour = np.zeros(mask.shape).astype(np.uint8)
    for k in range(len(mask)):
        contour[k] = get_contour(mask[k])

    # explicitly pass this with the function call
    """
    if com_high[1] > int((mask.shape[2] - 1) / 2):
        femur_left = False
    else:
        femur_left = True
    """

    correct_x = np.ndarray(0)
    correct_y = np.ndarray(0)
    correct_z = np.ndarray(0)
    correct_z_new = np.ndarray(0)
    # get coordinates for the sphere fitting
    if femur_left:
        for x in range(np.amin(contour_pts[2]), np.amax(contour_pts[2])+1):
            correct = np.nonzero(contour[layer_low:, :com_high[0], com_high[1]:])
            correct = np.nonzero(contour[layer_low:, :(x + com_high[0] - com_high[1]), x])
            correct_x = np.append(correct_x, np.full(correct[0].size, x))
            correct_y = np.append(correct_y, correct[1])
            correct_z = np.append(correct_z, (correct[0]+layer_low)*z_ratio)
            # correct_z_new = np.append(correct_z_new, (correct[0]+layer_low))
    else:
        for x in range(np.amin(contour_pts[2]), np.amax(contour_pts[2])+1):
            correct = np.nonzero(contour[layer_low:, :(-x+com_high[0]+com_high[1]), x])
            correct_x = np.append(correct_x, np.full(correct[0].size, x))
            correct_y = np.append(correct_y, correct[1])
            correct_z = np.append(correct_z, (correct[0]+layer_low)*z_ratio)
            # correct_z_new = np.append(correct_z_new, (correct[0]+layer_low))

    # get center coordinates of the fitting sphere
    r, x0, y0, z0 = sphere_fit(correct_x, correct_y, correct_z)
    # compensate pixel mm ratio between x, y and z axis
    z0 = z0/z_ratio
    center = round_to_int((z0[0], y0[0], x0[0]))

    # mark center in the mask
    if mark_points:
        mask[center[0], center[1], center[2]] = 5

    # find most distal layer with two areas and select the one below
    layer_selected = None
    n_pixels = 0
    gap_found = False
    # iterate through mask layers until there is a gap in the mask, then return the next layer that doesn't have a gap
    for n in range(mask.shape[0] - 1, 0, -1):
        l = len(measure.find_contours(mask[n], 0.8))
        p = np.count_nonzero(mask[n])
        if gap_found and (l == 1):
            layer_selected = n
            break
        else:
            if l > 1:  # or (p >= (n_pixels + 100)): # there is either a gap or a large increase in pixels
                gap_found = True

    if layer_selected is None:  # failsafe
        layer_selected = mask.shape[0] // 2

    if not reikeras:
        center_fn = find_femoral_neck_center(mask, femur_left, layer_selected)[1:]
    else:
        center_fn, layer_selected, reference_points = find_femoral_neck_center_reikeras(mask, layer_selected)
    print(f'layer_selected: {layer_selected}, reikeras is {reikeras}')

    if knee_mask is not None:

        km = np.copy(knee_mask)
        km, start_pt_orig, end_pt_orig, notch = calc_knee('femur', km, mark_points=True)

        lc_rc_vector = np.array(start_pt_orig)[1:] - np.array(end_pt_orig)[1:] # project everything onto one slice -> discard layer index

        line = bresenhamline([round_to_int(start_pt_orig[1:] - (1.5 * lc_rc_vector))], [round_to_int(end_pt_orig[1:] + (1.5 * lc_rc_vector))], -1)
        for u in range(len(line)):
            mask[layer_selected, int(line[u, 0]), int(line[u, 1])] = 5

    if mark_points:
        draw_sphere(mask, r, center, z_ratio=z_ratio)
        # draw two circles around the x-y-coords of the center
        # on the selected layer with the new radii in the mask
        draw_circle(mask, layer_selected, r, center[1:])
        draw_circle(mask, layer_selected, 5, center_fn)

        #draw_circle(mask, layer_selected - 1, r_1, center[1:])
        #draw_circle(mask, layer_selected - 1, r_2, center[1:])
        #draw_circle(mask, layer_selected - 1, r_3, center_fn)

        # draw reference line on the mask
        line = bresenhamline([(center[1], center[2])], center_fn, max_iter=-1)
        for u in range(len(line)):
            #mask[:, int(line[u, 0]), int(line[u, 1])] = 5
            mask[layer_selected, int(line[u, 0]), int(line[u, 1])] = 5

        if reikeras:
            line = bresenhamline([(reference_points[0][0], reference_points[0][1])], reference_points[2], max_iter=-1)
            for u in range(len(line)):
                #mask[:, int(line[u, 0]), int(line[u, 1])] = 5
                mask[layer_selected, int(line[u, 0]), int(line[u, 1])] = 5

            line = bresenhamline([(reference_points[1][0], reference_points[1][1])], reference_points[3], max_iter=-1)
            for u in range(len(line)):
                #mask[:, int(line[u, 0]), int(line[u, 1])] = 5
                mask[layer_selected, int(line[u, 0]), int(line[u, 1])] = 5

    if mark_points:
        mask = mask + contour
    
    if path_out is not None:
        write_image(mask, path_out)
    return mask, center, np.append(layer_selected, center_fn)


def calc_hip_(mask, z_ratio, path_out=None, path_out_ball=None):
    """
    calculates the center of the femur head and the reference line on hip level
    for the measurement of the antetorsion

    Parameters
    ----------
    path_in : str
        input path of the DICOM image file with the segmented femur on hip level
    path_out : str
        output path where the DICOM image file of the mask with the reference line
    path_out_ball : str

    Returns
    -------
    center : array of int
        3D coordinates of the center of the femur head
        -> start point of the reference line
    end : array of int
        2D coordinates of the possible endpoint of the reference line
    """

    for i in range(mask.shape[0]):
        tmp = np.argwhere(mask > 0)
        for j in range(min(tmp[:,1]), max(tmp[:,1])):
            for k in range(min(tmp[:,2]), max(tmp[:,2])):
                nonzero_neighbours = mask[i,j-1,k] + mask[i,j+1,k] + mask[i,j,k-1] + mask[i,j,k+1]
                if nonzero_neighbours > 2:
                    mask[i,j,k] = 1
        
        connected_components = cv2.connectedComponentsWithStats(mask[i].astype(np.uint8), connectivity=4)
        label_map = connected_components[1]
        component_sizes = connected_components[2][:, -1]
        large_components = np.argwhere(component_sizes > 50)
        mask_copy = np.zeros(mask[i].shape)
        for component in large_components:
            mask_copy += np.where(label_map == component[0], component[0], 0)
        
        mask_copy = np.where(mask_copy > 0, 1, 0)
        mask[i] = mask_copy

    # Calculate mask of femur
    # mask = get_mask(path_in)
    contour_pts = get_contour_points(mask)

    # get highest layer with a mask point, its centroid
    # and lowest layer with mask point on this centroid
    layer_high = np.amax(contour_pts[0])
    com_high = get_centroid(mask[layer_high-1])
    layer_low = layer_high
    while mask[layer_low-1, com_high[0], com_high[1]] != 0:
        layer_low -= 1

    # check whether the mask is a right or left femur
    if com_high[1] > int((mask.shape[2] - 1) / 2):
        femur_left = False
    else:
        femur_left = True

    # get contour of each layer of the mask separately
    contour = np.zeros(mask.shape).astype(np.uint8)
    for k in range(len(mask)):
        contour[k] = get_contour(mask[k])

    """
    # get coordinates for the sphere fitting
    if femur_left:
        correct = np.nonzero(contour[layer_low+1:, :com_high[0], com_high[1]:])
        correct_x = correct[2] + com_high[1]
        correct_y = correct[1]
        correct_z = (correct[0] + layer_low+1) * z_ratio
    else:
        correct = np.nonzero(contour[layer_low+1:, :com_high[0], :com_high[1]])
        correct_x = correct[2]
        correct_y = correct[1]
        correct_z = (correct[0] + layer_low+1) * z_ratio
    """

    correct_x = np.ndarray(0)
    correct_y = np.ndarray(0)
    correct_z = np.ndarray(0)
    correct_z_new = np.ndarray(0)
    # get coordinates for the sphere fitting
    if femur_left:
        for x in range(np.amin(contour_pts[2]), np.amax(contour_pts[2])+1):
            correct = np.nonzero(contour[layer_low:, :com_high[0], com_high[1]:])
            correct = np.nonzero(contour[layer_low:, :(x + com_high[0] - com_high[1]), x])
            correct_x = np.append(correct_x, np.full(correct[0].size, x))
            correct_y = np.append(correct_y, correct[1])
            correct_z = np.append(correct_z, (correct[0]+layer_low)*z_ratio)
            # correct_z_new = np.append(correct_z_new, (correct[0]+layer_low))
    else:
        for x in range(np.amin(contour_pts[2]), np.amax(contour_pts[2])+1):
            correct = np.nonzero(contour[layer_low:, :(-x+com_high[0]+com_high[1]), x])
            correct_x = np.append(correct_x, np.full(correct[0].size, x))
            correct_y = np.append(correct_y, correct[1])
            correct_z = np.append(correct_z, (correct[0]+layer_low)*z_ratio)
            # correct_z_new = np.append(correct_z_new, (correct[0]+layer_low))

    # get center coordinates of the fitting sphere
    r, x0, y0, z0 = sphere_fit(correct_x, correct_y, correct_z)
    # compensate pixel mm ratio between x, y and z axis
    z0 = z0/z_ratio
    center = round_to_int((z0[0], y0[0], x0[0]))

    # mark center in the mask
    mask[center[0], center[1], center[2]] = 5
    if path_out is not None:
        write_image(mask, path_out)

#    # find most distal layer with two areas and select the one below
#    for n in range(mask.shape[0]-1):
#        if len(measure.find_contours(mask[n], 0.8)) > 1:
#            d = measure.find_contours(mask[n], 0.8)
#            layer_selected = n-1
#            break

    for n in range(mask.shape[0]-1, 0, -1):
        if len(measure.find_contours(mask[n], 0.8)) == 1 and \
                pts_on_circle(mask[n], r*2, [center[1], center[2]]):
            layer_selected = n
            break

    mask_new, contour, r_1, r_2 = contour_femoral_neck(mask, contour, layer_selected, center, r)
    while np.count_nonzero(mask_new == 1) < 65:
        layer_selected = layer_selected - 1
        mask_new, contour, r_1, r_2 = contour_femoral_neck(mask, contour, layer_selected, center, r)

    # center of femoral neck
    center_fn = get_centroid(mask_new)

    # get coordinates of the selected contour points
    contour_pts_l = np.nonzero(contour[layer_selected])

    distance_center = (contour_pts_l[0]-center_fn[0])**2 + (contour_pts_l[1]-center_fn[1])**2
    r_3 = math.sqrt(np.median(distance_center)) * 1.5

    # get contour also only with small distance to center_fn on the selected layer
    for w in range(mask.shape[2] - 1):  # all x coordinates
        if r_3 ** 2 - (w - center_fn[1]) ** 2 >= 0:
            y_b = int(round(math.sqrt(r_3 ** 2 - (w - center_fn[1]) ** 2)))
            if center_fn[0] + y_b < mask.shape[1]:
                contour[layer_selected, center_fn[0] + y_b:, w] = 0
            if center_fn[0] - y_b > 0:
                contour[layer_selected, :center_fn[0] - y_b, w] = 0
        else:
            contour[layer_selected, :, w] = 0

    contour_pts_l = np.nonzero(contour[layer_selected])

    # get index of the contour point after the largest gap
    # -> first contour point on the other side of the mask
    diff = np.ediff1d(contour_pts_l[0])
    ind_gap = np.argsort(diff)[-1] + 1

    def g(x, m):
        return m * x

    if diff.max() == 1:
        # lsf through the center and the contour points to get its slope
        popt, _ = curve_fit(g, contour_pts_l[1] - center[2],
                            contour_pts_l[0] - center[1])
        m_new = popt[0]
    else:
        # separated lsf through the center and the contour points on both sides
        # to get their slopes and calculate the mean
        popt1, _ = curve_fit(g, contour_pts_l[1][:ind_gap] - center[2],
                             contour_pts_l[0][:ind_gap] - center[1])
        popt2, _ = curve_fit(g, contour_pts_l[1][ind_gap:] - center[2],
                             contour_pts_l[0][ind_gap:] - center[1])
        # m_new = np.mean([popt1[0], popt2[0]])
        m_new = np.tan((np.arctan(popt2[0]) + np.arctan(popt1[0])) / 2)

    # find endpoint for the reference line
    if femur_left:
        end = round_to_int(((-80) * m_new + center[1], -80 + center[2]))
    else:
        end = round_to_int((80 * m_new + center[1], 80 + center[2]))
        if end[0]>mask.shape[1] or end[1]>mask.shape[2]:
            end = round_to_int((50 * m_new + center[1], 50 + center[2]))

    # draw the fitted sphere
    draw_sphere(mask, r, center, z_ratio=z_ratio)
    # draw two circles around the x-y-coords of the center
    # on the selected layer with the new radii in the mask
    draw_circle(mask, layer_selected, r_1, center[1:])
    draw_circle(mask, layer_selected, r_2, center[1:])
    draw_circle(mask, layer_selected, r_3, center_fn)

    draw_circle(mask, layer_selected-1, r_1, center[1:])
    draw_circle(mask, layer_selected-1, r_2, center[1:])
    draw_circle(mask, layer_selected-1, r_3, center_fn)

    # draw reference line on the mask
    line = bresenhamline([(center[1], center[2])], end, max_iter=-1)
    for u in range(len(line)):
        mask[:, int(line[u, 0]), int(line[u, 1])] = 5

    if path_out is not None:
        write_image(mask+contour, path_out)

    return mask+contour, center, np.append(layer_selected, end)


