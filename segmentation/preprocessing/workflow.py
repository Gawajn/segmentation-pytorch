from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
from segmentation.settings import  Preprocessingfunction
from segmentation.util import rgb2gray, gray_to_rgb, show_images, gray_to_rgb_luminance

import albumentations as albu
import numpy as np

from segmentation.binarization.doxapy_bin import binarize, BinarizationAlgorithm

@dataclass
class PreprocessingTransforms:
    input_transform: Optional[albu.Compose] = None
    aug_transform: Optional[albu.Compose] = None
    tta_transform: Optional[albu.Compose] = None
    post_transforms: Optional[albu.Compose] = None
    lambda_transforms: Optional[Dict] = None

    def to_dict(self):
        return {
            "input_transform": albu.to_dict(self.input_transform) if self.input_transform else None,
            "aug_transform": albu.to_dict(self.aug_transform) if self.aug_transform else None,
            "tta_transform": albu.to_dict(self.tta_transform) if self.tta_transform else None,
            "post_transforms": albu.to_dict(self.post_transforms) if self.post_transforms else None
        }

    def transform_train(self, image, mask) -> Dict:
        res = {"image": image, "mask": mask}
        if self.input_transform:
            res = self.input_transform(**res)
        if self.aug_transform:
            res = self.aug_transform(**res)
        if self.post_transforms:
            res = self.post_transforms(**res)
        return res

    def transform_predict(self, image) -> Dict:
        res = {"image": image}
        if self.input_transform:
            res = self.input_transform(**res)
        if self.tta_transform:
            res = self.tta_transform(**res)
        if self.post_transforms:
            res = self.post_transforms(**res)
        return res

    @staticmethod
    def from_dict(d: Dict, lambda_transforms: Dict[str, Callable] = None) -> 'PreprocessingTransforms':
        return PreprocessingTransforms(
            input_transform=albumentations.from_dict(d["input_transform"]) if d["input_transform"] is not None else None,
            aug_transform=albumentations.from_dict(d["aug_transform"]) if d["aug_transform"] is not None else None,
            tta_transform=albumentations.from_dict(d["tta_transform"]) if d["tta_transform"] is not None else None,
            post_transforms=albumentations.from_dict(d["post_transforms"]) if d["post_transforms"] is not None else None,
            lambda_transforms=lambda_transforms
        )

    def get_train_transforms(self):
        return PreprocessingTransforms(
            input_transform=self.input_transform,
            aug_transform=self.aug_transform,
            post_transforms=self.post_transforms,
            lambda_transforms=self.lambda_transforms
        )

    def get_test_transforms(self):
        return PreprocessingTransforms(
            input_transform=self.input_transform,
            tta_transform=self.tta_transform,
            post_transforms=self.post_transforms,
            lambda_transforms=self.lambda_transforms
        )

import albumentations.core.transforms_interface


class ColorMapTransform(albu.core.transforms_interface.BasicTransform):
    def __init__(self, color_map: Dict[int, Tuple[int]], **params):
        super().__init__(always_apply=True)
        self.color_map = color_map

    def color_to_label(self,mask):
        out = np.zeros(mask.shape[0:2], dtype=np.int32)

        if mask.ndim == 2:
            return mask.astype(np.int32) / 255

        if mask.shape[2] == 2:
            return mask[:, :, 0].astype(np.int32) / 255
        mask = mask.astype(np.uint32)
        mask = 256 * 256 * mask[:, :, 0] + 256 * mask[:, :, 1] + mask[:, :, 2]
        for label, color in self.color_map.items():
            color_1d = 256 * 256 * color[0] + 256 * color[1] + color[2]
            out += np.int32((mask == color_1d) * label)
        return out

    def apply_to_mask(self, mask, **params):
        if self.color_map:
            if mask.ndim == 3:
                return self.color_to_label(mask)
            elif mask.ndim == 2:
                u_values = np.unique(mask)
                mask2 = mask
                for ind, x in enumerate(u_values):
                    mask2[mask == x] = ind
                return mask2
        else:
            raise "No Colormap specified"

    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "mask": self.apply_to_mask,
        }

    def get_transform_init_args_names(self):
        return ["color_map"]


class GrayToRGBTransform(albu.core.transforms_interface.BasicTransform):
    def __init__(self, **params):
        super().__init__(always_apply=True)

    def apply_to_image(self, image, **params):
        return gray_to_rgb(image)

    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply_to_image,
        }

    def get_transform_init_args_names(self):
        return ()


class BinarizeGauss(albu.core.transforms_interface.BasicTransform):
    def __init__(self, **params):
        super().__init__(**params)

    def apply_to_image(self, image, **params):
        from segmentation.preprocessing.basic_binarizer import gauss_threshold

        image = rgb2gray(image).astype(np.uint8)
        return gray_to_rgb(gauss_threshold(image))

    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply_to_image,
        }

    def get_transform_init_args_names(self):
        return ()
class BinarizeDoxapy(albu.core.transforms_interface.BasicTransform):
    def __init__(self, algorithm=BinarizationAlgorithm.ISauvola.value, **params):
        super().__init__(**params)
        self.algorithm = algorithm
    def apply_to_image(self, image, **params):
        from segmentation.binarization.doxapy_bin import binarize, BinarizationAlgorithm

        binary = binarize(image,algorithm=BinarizationAlgorithm(self.algorithm)).astype("uint8") * 255

        return gray_to_rgb(binary)

    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply_to_image,
        }

    def get_transform_init_args_names(self):
        return ["algorithm"]

class NetworkEncoderTransform(albu.core.transforms_interface.BasicTransform):
    def __init__(self, preprocessing_function: str, **params):
        super().__init__(always_apply=True)
        self.preprocessing_function = preprocessing_function

    def apply_to_image(self, image, **params):
        fn = Preprocessingfunction(self.preprocessing_function).get_preprocessing_function()
        return fn(image)

    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply_to_image,
        }

    def get_transform_init_args_names(self):
        return ["preprocessing_function"]

