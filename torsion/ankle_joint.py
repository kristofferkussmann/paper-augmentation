from skimage.measure import regionprops, label
import numpy as np
from . import get_centroid, write_image, bresenhamline


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
