import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy
numpy.random.seed(0)

import sys
import os
import ctypes

def increase_stack_size(size):
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadStackGuarantee(ctypes.byref(ctypes.c_ulong(size)))
    else:
        import resource
        resource.setrlimit(resource.RLIMIT_STACK, (size, resource.RLIM_INFINITY))

# Increase stack size to 8 MB (or as needed)
increase_stack_size(8 * 1024 * 1024)

# sys.path.append('/home/simon/Work/mri-augmentation/Code')
sys.path.append('C:/Users/krist/paper_augmentation/paper-augmentation')

from torsion import calc_hip, calc_knee, calc_ankle_joint, calc_pma, calc_ccd
import numpy as np
import SimpleITK as sitk
from skimage.measure import label
import json
import argparse
from pathlib import Path

#import resource
#resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


def trace(frame, event, arg):
    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace


def get_largest_CC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1).astype(np.uint16)
    return largestCC


def main(file, aug):
    """
    Calculate torsion angles for femur and tibia. Save reference lines as segmentation masks with overlayed reference lines. Save torsion angles in json file.
    All output files are written to the input directory.

    :param file: Path to directory containing segmentation masks hip_seg.nii.gz, knee_seg.nii.gz, ankle_seg.nii.gz
    :param aug: Augmentation type (baseline, default, mr)
    """
    i = file

    tors_hl = np.nan
    tors_hr = np.nan
    tors_kfl = np.nan
    tors_kfr = np.nan
    tors_ktl = np.nan
    tors_ktr = np.nan
    tors_al = np.nan
    tors_ar = np.nan

    hip_seg = sitk.ReadImage(str(i / ('hip_seg.nii.gz' if aug == 'mr' else f'hip_seg_{aug}.nii.gz')))
    knee_seg = sitk.ReadImage(str(i / ('knee_seg.nii.gz' if aug == 'mr' else f'knee_seg_{aug}.nii.gz')))
    ankle_seg = sitk.ReadImage(str(i / ('ankle_seg.nii.gz' if aug == 'mr' else f'ankle_seg_{aug}.nii.gz')))

    values = dict()

    # HIP
    spacing = hip_seg.GetSpacing()
    z_ratio = abs(spacing[2]) / 2 * abs(spacing[0])  # /2 bei Pat 48-64

    hip_array = sitk.GetArrayFromImage(hip_seg)
    hip_l = hip_array[:, :, :int(hip_array.shape[2] / 2)]
    hip_r = hip_array[:, :, int(hip_array.shape[2] / 2):]

    try:
        # p1 is the center of femoral head, p2 is the endpoint of the reference line
        mask_hr, p1_hr, p2_hr = calc_hip(get_largest_CC(hip_r), z_ratio=z_ratio, femur_left=False)
        mask_hl, p1_hl, p2_hl = calc_hip(get_largest_CC(hip_l), z_ratio=z_ratio, femur_left=True)

        tors_hr = np.arctan((p2_hr[1] - p1_hr[1]) / (p2_hr[2] - p1_hr[2])) * 180 / np.pi
        tors_hl = np.arctan((p2_hl[1] - p1_hl[1]) / (p1_hl[2] - p2_hl[2])) * 180 / np.pi

        values['hip_right'] = tors_hl  # swap left and right because right image side is left patient side
        values['hip_left'] = tors_hr

        mask_hip = np.concatenate((mask_hl, mask_hr), axis=2)

        mask_hip = sitk.GetImageFromArray(mask_hip)
        mask_hip.CopyInformation(hip_seg)
        mask_hip.SetSpacing(hip_seg.GetSpacing())
        mask_hip.SetOrigin(hip_seg.GetOrigin())
        mask_hip.SetDirection(hip_seg.GetDirection())
        sitk.WriteImage(mask_hip, str(i / f'hip_ref_{aug}.nii.gz'))
    except (TypeError, UnboundLocalError, IndexError, ValueError, AssertionError) as e:
        print(f'Calc hip failed for {i}, {e}')

    # KNEE FEMUR (caculation of the CCD angle is also run in this section)
    knee_array = sitk.GetArrayFromImage(knee_seg)
    knee_l = knee_array[:, :, :int(knee_array.shape[2] / 2)]
    knee_r = knee_array[:, :, int(knee_array.shape[2] / 2):]

    knee_l = np.where(knee_l == 1, 1, 0)  # set tibia to 0
    knee_r = np.where(knee_r == 1, 1, 0)

    try:
        mask_kfr, p1_kfr, p2_kfr, _ = calc_knee('femur', get_largest_CC(knee_r))
        mask_kfl, p1_kfl, p2_kfl, _ = calc_knee('femur', get_largest_CC(knee_l))

        if p2_kfr[2] > p1_kfr[2]:
            tors_kfr = np.arctan((p1_kfr[1] - p2_kfr[1]) / (p2_kfr[2] - p1_kfr[2])) * 180 / np.pi
        else:
            tors_kfr = np.arctan((p2_kfr[1] - p1_kfr[1]) / (p1_kfr[2] - p2_kfr[2])) * 180 / np.pi
        if p2_kfl[2] > p1_kfl[2]:
            tors_kfl = np.arctan((p2_kfl[1] - p1_kfl[1]) / (p2_kfl[2] - p1_kfl[2])) * 180 / np.pi
        else:
            tors_kfl = np.arctan((p1_kfl[1] - p2_kfl[1]) / (p1_kfl[2] - p2_kfl[2])) * 180 / np.pi

        # calculate caput-collum-diaphyseal angle
        mask_ccd_r, ccd_r = calc_ccd(get_largest_CC(hip_r), get_largest_CC(knee_r))
        mask_ccd_l, ccd_l = calc_ccd(get_largest_CC(hip_l), get_largest_CC(knee_l))

        values['knee_femur_right'] = tors_kfl  # swap left and right because right image side is left patient side
        values['knee_femur_left'] = tors_kfr

        values['femur_right'] = tors_hl + tors_kfl
        values['femur_left'] = tors_hr + tors_kfr

        values['ccd_right'] = ccd_l
        values['ccd_left'] = ccd_r

        mask_knee = np.concatenate((mask_kfl, mask_kfr), axis=2)
        mask_knee = sitk.GetImageFromArray(mask_knee)
        mask_knee.CopyInformation(knee_seg)
        mask_knee.SetSpacing(knee_seg.GetSpacing())
        mask_knee.SetOrigin(knee_seg.GetOrigin())
        mask_knee.SetDirection(knee_seg.GetDirection())
        sitk.WriteImage(mask_knee, str(i / f'knee_femur_ref_{aug}.nii.gz'))

        mask_ccd = np.concatenate((mask_ccd_l, mask_ccd_r), axis=2)
        mask_ccd = sitk.GetImageFromArray(mask_ccd)
        #mask_ccd.CopyInformation(knee_seg)
        mask_ccd.SetSpacing(knee_seg.GetSpacing())
        mask_ccd.SetOrigin(knee_seg.GetOrigin())
        mask_ccd.SetDirection(knee_seg.GetDirection())
        sitk.WriteImage(mask_ccd, str(i / f'hip_knee_femur_ref_ccd_{aug}.nii.gz'))
    except (TypeError, UnboundLocalError, IndexError, ValueError, AssertionError) as e:
        print(f'Calc knee/f failed for {i}, {e}')

    # KNEE TIBIA
    knee_l = knee_array[:, :, :int(knee_array.shape[2] / 2)]
    knee_r = knee_array[:, :, int(knee_array.shape[2] / 2):]

    knee_l = np.where(knee_l == 2, 1, 0)  # set femur to 0
    knee_r = np.where(knee_r == 2, 1, 0)

    try:
        mask_ktr, p1_ktr, p2_ktr, _ = calc_knee('tibia', get_largest_CC(knee_r))
        mask_ktl, p1_ktl, p2_ktl, _ = calc_knee('tibia', get_largest_CC(knee_l))

        if p2_ktr[2] > p1_ktr[2]:
            tors_ktr = np.arctan((p1_ktr[1] - p2_ktr[1]) / (p2_ktr[2] - p1_ktr[2])) * 180 / np.pi
        else:
            tors_ktr = np.arctan((p2_ktr[1] - p1_ktr[1]) / (p1_ktr[2] - p2_ktr[2])) * 180 / np.pi
        if p2_ktl[2] > p1_ktl[2]:
            tors_ktl = np.arctan((p2_ktl[1] - p1_ktl[1]) / (p2_ktl[2] - p1_ktl[2])) * 180 / np.pi
        else:
            tors_ktl = np.arctan((p1_ktl[1] - p2_ktl[1]) / (p1_ktl[2] - p2_ktl[2])) * 180 / np.pi

        values['knee_tibia_right'] = tors_ktl  # swap left and right because right image side is left patient side
        values['knee_tibia_left'] = tors_ktr

        mask_knee = np.concatenate((mask_ktl, mask_ktr), axis=2)
        mask_knee = sitk.GetImageFromArray(mask_knee)
        mask_knee.CopyInformation(knee_seg)
        mask_knee.SetSpacing(knee_seg.GetSpacing())
        mask_knee.SetOrigin(knee_seg.GetOrigin())
        mask_knee.SetDirection(knee_seg.GetDirection())
        sitk.WriteImage(mask_knee, str(i / f'knee_tibia_ref_{aug}.nii.gz'))
    except (TypeError, UnboundLocalError, IndexError, ValueError, AssertionError) as e:
        print(f'Calc knee/t failed for {i}, {e}')

    # ANKLE
    ankle_array = sitk.GetArrayFromImage(ankle_seg)
    ankle_l = ankle_array[:, :, :int(ankle_array.shape[2] / 2)]
    ankle_r = ankle_array[:, :, int(ankle_array.shape[2] / 2):]

    ankle_tibia_l = np.where(ankle_l == 1, 1, 0)  # set fibula to 0
    ankle_tibia_r = np.where(ankle_r == 1, 1, 0)

    ankle_fibula_l = np.where(ankle_l == 2, 1, 0)  # set tibia to 0
    ankle_fibula_r = np.where(ankle_r == 2, 1, 0)

    try:
        mask_ar, p1_ar, p2_ar = calc_ankle_joint(get_largest_CC(ankle_tibia_r), get_largest_CC(ankle_fibula_r))
        mask_al, p1_al, p2_al = calc_ankle_joint(get_largest_CC(ankle_tibia_l), get_largest_CC(ankle_fibula_l))

        # calculate torsion
        tors_ar = np.arctan((p2_ar[1] - p1_ar[1]) / (p2_ar[2] - p1_ar[2])) * 180 / np.pi
        tors_al = np.arctan((p2_al[1] - p1_al[1]) / (p1_al[2] - p2_al[2])) * 180 / np.pi

        # calculate plafond malleolus angle
        mask_pma_ar, pma_r = calc_pma(get_largest_CC(ankle_tibia_r), get_largest_CC(ankle_fibula_r))
        mask_pma_al, pma_l = calc_pma(get_largest_CC(ankle_tibia_l), get_largest_CC(ankle_fibula_l))

        values['ankle_right'] = tors_al  # swap left and right because right image side is left patient side
        values['ankle_left'] = tors_ar

        values['tibia_right'] = tors_ktl + tors_al
        values['tibia_left'] = tors_ktr + tors_ar

        values['pma_right'] = pma_l
        values['pma_left'] = pma_r

        mask_ankle = np.concatenate((mask_al, mask_ar), axis=2)
        mask_ankle = sitk.GetImageFromArray(mask_ankle)
        mask_ankle.CopyInformation(ankle_seg)
        mask_ankle.SetSpacing(ankle_seg.GetSpacing())
        mask_ankle.SetOrigin(ankle_seg.GetOrigin())
        mask_ankle.SetDirection(ankle_seg.GetDirection())
        sitk.WriteImage(mask_ankle, str(i / f'ankle_ref_{aug}.nii.gz'))

        mask_ankle_pma = np.concatenate((mask_pma_al, mask_pma_ar), axis=2)
        mask_ankle_pma = sitk.GetImageFromArray(mask_ankle_pma)
        mask_ankle_pma.CopyInformation(ankle_seg)
        mask_ankle_pma.SetSpacing(ankle_seg.GetSpacing())
        mask_ankle_pma.SetOrigin(ankle_seg.GetOrigin())
        mask_ankle_pma.SetDirection(ankle_seg.GetDirection())
        sitk.WriteImage(mask_ankle_pma, str(i / f'ankle_ref_pma_{aug}.nii.gz'))
    except (TypeError, UnboundLocalError, IndexError, ValueError, AssertionError) as e:
        print(f'Calc ankle failed for {i}, {e}')

    with open(i / f'values_{aug}.json', 'w') as f:
        json.dump(values, f)


if __name__ == '__main__':
    """
    How to use:
    python torsion_detection_nnunet.py -aug [baseline, default, mr] -file [/path/to/file]
    baseline for no augmentation, default for default nnUNet augmentation, mr for MRI-specific augmentation
    /path/to/file should point to a directory containing the segmentation masks hip_seg.nii.gz, knee_seg.nii.gz, ankle_seg.nii.gz. For an example, refer to the torsion_format directory in the project data repository (see readme file).
    For batch processing, use the run_torsion_detection.py script.
    """

    #example call: python torsion_detection_nnunet.py -aug mr -file "C:\Users\krist\OneDrive - Students RWTH Aachen University\studium\hiwi\hiwi_uka\data_uka_angle_calc\augmentation-paper\torsion_format\ankle\reference\Patient_2"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-aug', type=str, default='mr', help='augmentation type')
    parser.add_argument('-file', type=str, help='file')
    args = parser.parse_args()
    file = Path(args.file)
    main(file, args.aug)
