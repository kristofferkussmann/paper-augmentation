import random
from typing import Any
import numpy as np
from skimage.transform import resize, rotate
from torchio.transforms.augmentation.intensity import RandomMotion, RandomGhosting, RandomSpike, RandomBiasField, \
    RandomBlur, RandomNoise, RandomGamma
from torchio.transforms import ZNormalization
import torchio as tio
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class ResizeTransform(AbstractTransform):
    def __init__(self, size, data_key="data", label_key="seg", p_per_sample=.5):
        self.size = size
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        # data dict value is a numpy array (b, c, z, y, x), where b is the batch size, c is the number of channels
        print(data_dict[self.data_key].shape)
        print(data_dict[self.label_key].shape)
        for b in range(len(data_dict[self.data_key])):  # for each batch
            if np.random.uniform() < self.p_per_sample:
                for c in range(data_dict[self.data_key][b].shape[0]):  # for each channel
                    tz, ty, tx = self.size
                    data_dict[self.data_key][b][c] = resize(data_dict[self.data_key][b][c], (tz, ty, tx),
                                                            preserve_range=True)
                    data_dict[self.label_key][b][c] = resize(data_dict[self.label_key][b][c], (tz, ty, tx),
                                                             anti_aliasing=False,
                                                             order=0, preserve_range=True)
        return data_dict


class RandomCropTransform(AbstractTransform):
    def __init__(self, size, data_key="data", label_key="seg", p_per_sample=.5):
        self.size = size
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                for c in range(data_dict[self.data_key][b].shape[0]):
                    z, y, x = data_dict[self.data_key][b].shape[1:]  # exclude channel
                    tz, ty, tx = self.size
                    if z == tz and y == ty and x == tx:
                        continue
                    if z < tz or y < ty or x < tx:
                        data_dict[self.data_key][b][c] = resize(data_dict[self.data_key][b][c], (tz, ty, tx),
                                                                preserve_range=True)
                        data_dict[self.label_key][b][c] = resize(data_dict[self.label_key][b][c], (tz, ty, tx),
                                                                 anti_aliasing=False,
                                                                 order=0, preserve_range=True)
                        continue

                    z1 = random.randint(0, z - tz)
                    y1 = random.randint(0, y - ty)
                    x1 = random.randint(0, x - tx)
                    data_dict[self.data_key][b][c] = data_dict[self.data_key][b][c][z1: z1 + tz, y1: y1 + ty,
                                                     x1: x1 + tx]
                    data_dict[self.label_key][b][c] = data_dict[self.label_key][b][c][z1: z1 + tz, y1: y1 + ty,
                                                      x1: x1 + tx]
        return data_dict


class RandomRotateTransform(AbstractTransform):
    def __init__(self, degrees, data_key="data", label_key="seg", p_per_sample=.5):
        self.degrees = degrees
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                rotate_degree = random.randint(self.degrees[0], self.degrees[1])
                for c in range(data_dict[self.data_key][b].shape[0]):
                    data_dict[self.data_key][b][c] = rotate(data_dict[self.data_key][b][c], rotate_degree,
                                                            preserve_range=True)
                    data_dict[self.label_key][b][c] = rotate(data_dict[self.label_key][b][c], rotate_degree,
                                                             preserve_range=True, order=0)

        return data_dict


class RandomFlipTransform(AbstractTransform):
    def __init__(self, axis, data_key="data", label_key="seg", p_per_sample=.5):
        self.axis = axis
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                for c in range(data_dict[self.data_key][b].shape[0]):
                    if self.axis == 'horizontal':
                        for z in range(data_dict[self.data_key][b][c].shape[0]):
                            data_dict[self.data_key][b][c][z] = np.fliplr(data_dict[self.data_key][b][c][z])
                            data_dict[self.label_key][b][c][z] = np.fliplr(data_dict[self.label_key][b][c][z])
                    elif self.axis == 'vertical':
                        for z in range(data_dict[self.data_key][b][c].shape[0]):
                            data_dict[self.data_key][b][c][z] = np.flipud(data_dict[self.data_key][b][c][z])
                            data_dict[self.label_key][b][c][z] = np.flipud(data_dict[self.label_key][b][c][z])
                    else:
                        raise ValueError("Invalid axis. Must be 'horizontal' or 'vertical'.")

        return data_dict


class RandomNoiseTransform(AbstractTransform):
    def __init__(self, mean=0, std=(0, 0.25), data_key="data", label_key="seg", p_per_sample=.5):
        self.mean = mean
        self.std = std
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                for c in range(data_dict[self.data_key][b].shape[0]):
                    data_dict[self.data_key][b][c] = RandomNoise(mean=self.mean, std=self.std)(ZNormalization())(
                        data_dict[self.data_key][b][c])
                    # data_dict[self.label_key][b][c] = RandomNoise(mean=self.mean, std=self.std)(ZNormalization())(
                       # data_dict[self.label_key][b][c])

        return data_dict


class RandomSpikeTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=0.02):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            random_spike = RandomSpike(num_spikes=(0, 3), intensity=(0, 1.5), p=self.p_per_sample)
            data_dict[self.data_key][b] = random_spike(data_dict[self.data_key][b])
            # data_dict[self.label_key][b] = random_spike(data_dict[self.label_key][b])
        return data_dict


class RandomBlurTransform(AbstractTransform):
    def __init__(self, std=(0, 0.8), data_key="data", label_key="seg", p_per_sample=0.5):
        self.std = std
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            random_blur = RandomBlur(std=self.std, p=self.p_per_sample)
            data_dict[self.data_key][b] = random_blur(data_dict[self.data_key][b])
            # data_dict[self.label_key][b] = random_blur(data_dict[self.label_key][b])
        return data_dict


class RandomBiasFieldTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=0.5):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            random_bias_field = RandomBiasField(coefficients=(0, 0.5), order=1, p=self.p_per_sample)
            data_dict[self.data_key][b] = random_bias_field(data_dict[self.data_key][b])
            # data_dict[self.label_key][b] = random_bias_field(data_dict[self.label_key][b])
        return data_dict


class RandomMotionTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=0.5):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            random_motion = RandomMotion(degrees=(-4, 4), translation=(-8, 8), num_transforms=np.random.randint(1, 2),
                                         image_interpolation='linear', p=self.p_per_sample)
            data_dict[self.data_key][b] = random_motion(data_dict[self.data_key][b])
            # data_dict[self.label_key][b] = random_motion(data_dict[self.label_key][b])
        return data_dict


class RandomGhostingTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=0.5):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            random_ghosting = RandomGhosting(num_ghosts=(0, 5), axes=1, intensity=(0, 0.5), p=self.p_per_sample)
            data_dict[self.data_key][b] = random_ghosting(data_dict[self.data_key][b])
            # data_dict[self.label_key][b] = random_ghosting(data_dict[self.label_key][b])
        return data_dict


class ConstantArtefactsTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            for c in range(len(data_dict[self.data_key][b].shape[0])):
                data_dict[self.data_key][b][c] = tio.Compose(
                    [RandomMotion(degrees=(3, 3), translation=(4, 4), num_transforms=2, image_interpolation='linear'),
                     RandomGhosting(num_ghosts=(3, 3), axes=1, intensity=(0.3, 0.3))])(data_dict[self.data_key][b][c])
                data_dict[self.label_key][b][c] = tio.Compose(
                    [RandomMotion(degrees=(3, 3), translation=(4, 4), num_transforms=2, image_interpolation='linear'),
                     RandomGhosting(num_ghosts=(3, 3), axes=1, intensity=(0.3, 0.3))])(data_dict[self.label_key][b][c])
        return data_dict
