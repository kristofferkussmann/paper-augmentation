from typing import Union, Tuple, List, Any

from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.motion_augments import ResizeTransform, RandomCropTransform, \
    RandomRotateTransform, RandomFlipTransform, RandomNoiseTransform, RandomSpikeTransform, RandomBiasFieldTransform, \
    RandomBlurTransform, RandomMotionTransform, RandomGhostingTransform
import random
import numpy as np
import torch


class nnUNetTrainerCustom(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), aug: str = 'no', max_epochs: int = 150,
                 test: bool = False):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.aug = aug
        self.num_epochs = max_epochs
        self.test = test

        print(f'Test mode is {self.test}')

    def get_training_transforms(
            self,
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: dict,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            order_resampling_data: int = 3,
            order_resampling_seg: int = 1,
            border_val_seg: int = -1,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None
    ) -> AbstractTransform:
        # this is default
        tr_transforms = []

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        if self.aug == 'no':  # no augmentation means we stop here
            tr_transforms.append(SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=None,
                do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
                do_rotation=False, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'],
                angle_z=rotation_for_DA['z'],
                p_rot_per_axis=1,  # todo experiment with this
                do_scale=False, scale=(0.7, 1.4),
                border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
                border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
                random_crop=False,  # random cropping is part of our dataloaders
                p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
                independent_scale_for_each_axis=False  # todo experiment with this
            ))

            if do_dummy_2d_data_aug:
                tr_transforms.append(Convert2DTo3DTransform())

            tr_transforms.append(RemoveLabelTransform(-1, 0))
            tr_transforms.append(RenameTransform('seg', 'target', True))

            if deep_supervision_scales is not None:
                tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                                  output_key='target'))

            tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
            tr_transforms = Compose(tr_transforms)
            return tr_transforms

        # this has cropping, rotation, scaling
        tr_transforms.append(SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,  # todo experiment with this
            do_scale=True, scale=(0.7, 1.4),
            border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
            random_crop=False,  # random cropping is part of our dataloaders
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False  # todo experiment with this
        ))

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())



        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))

        # Take blur out because we want this in the MR augmentation
        """
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
        """

        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=ignore_axes))
        tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

        # this is flipping
        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                               mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))


        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            tr_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(foreground_labels), 0)),
                p_per_sample=0.4,
                key="data",
                strel_size=(1, 8),
                p_per_label=1))
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15))

        if self.aug == 'default':  # default augmentation means we stop here
            tr_transforms.append(RenameTransform('seg', 'target', True))

            if deep_supervision_scales is not None:
                tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                                  output_key='target'))

            tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))

            tr_transforms = Compose(tr_transforms)
            return tr_transforms

        if self.aug != 'mr':
            raise ValueError(f'Invalid augmentation type: {self.aug}, must be one of "no", "default", "mr"')

        # this is the MR augmentation
        tr_transforms.append(RandomBlurTransform())
        tr_transforms.append(RandomSpikeTransform())
        tr_transforms.append(RandomBiasFieldTransform())

        tr_transforms.append(RandomMotionTransform())
        tr_transforms.append(RandomGhostingTransform())

        tr_transforms.append(RenameTransform('seg', 'target', True))

        # this must be at the bottom, not sure what it does
        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                       if ignore_label is not None else regions,
                                                                       'target', 'target'))

        if deep_supervision_scales is not None:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))

        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))

        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    """
    def get_validation_transforms(
            self,
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))

        # this is custom
        augmentation_type = self.aug
        if (augmentation_type == 'no') or self.test:  # don't augment when we are testing, as test images have artefacts
            pass
        elif augmentation_type == 'basic':
            # tr_transforms.append(RandomCropTransform(size=(40, 196, 348)))
            val_transforms.append(RandomRotateTransform(degrees=(-15, 15)))
            val_transforms.append(RandomFlipTransform(axis=random.choice(['horizontal', 'vertical'])))
            # tr_transforms.append(RandomNoiseTransform())
            val_transforms.append(GaussianNoiseTransform(p_per_sample=0.5))
        elif augmentation_type == 'advanced':  # do basic + advanced
            val_transforms.append(RandomRotateTransform(degrees=(-15, 15)))
            val_transforms.append(RandomFlipTransform(axis=random.choice(['horizontal', 'vertical'])))
            val_transforms.append(GaussianNoiseTransform(p_per_sample=0.5))

            val_transforms.append(RandomBlurTransform())
            val_transforms.append(RandomSpikeTransform())
            val_transforms.append(RandomBiasFieldTransform())
        elif augmentation_type == 'motion':  # do basic + advanced + motion
            val_transforms.append(RandomRotateTransform(degrees=(-15, 15)))
            val_transforms.append(RandomFlipTransform(axis=random.choice(['horizontal', 'vertical'])))
            val_transforms.append(GaussianNoiseTransform(p_per_sample=0.5))

            val_transforms.append(RandomBlurTransform())
            val_transforms.append(RandomSpikeTransform())
            val_transforms.append(RandomBiasFieldTransform())

            val_transforms.append(RandomMotionTransform())
            val_transforms.append(RandomGhostingTransform())
        else:
            raise ValueError(
                f'Invalid augmentation type: {augmentation_type}, must be one of "no", "basic", "advanced", "motion"')

        if is_cascaded:
            val_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))

        val_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            val_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                        if ignore_label is not None else regions,
                                                                        'target', 'target'))

        if deep_supervision_scales is not None:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms
        """
