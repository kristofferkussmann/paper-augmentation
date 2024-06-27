import sys
sys.path.append('C:/Users/krist/paper_augmentation/paper-augmentation/torsion')

from bresenham_slope import bresenhamline
from mask import get_mask, get_centroid, write_image
from hip import calc_hip
from knee import calc_knee
from ankle_joint import calc_ankle_joint, calc_pma