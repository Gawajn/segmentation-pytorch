import enum
from typing import Union

import PIL
from PIL import Image

from segmentation.binarization.doxapy_bin import BinarizationAlgorithm, BinarizationParams, _needs_binarization, \
    binarize
from segmentation.datasets.dataset import get_rescale_factor

import numpy as np


class RescaleMethod(enum.Enum):
    MASK = "mask"
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"

    def get_pil_scale(self):
        return {
            "mask" : PIL.Image.NEAREST,
            "nearest": PIL.Image.NEAREST,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS
        }[self.value]


class SourceImage:
    fail_on_binarize = False

    @staticmethod
    def load(filename) -> 'SourceImage':
        img = Image.open(filename)
        if img.mode == "CMYK":
            img = img.convert("RGB")
        elif img.mode == "RGBA":
            img = img.convert("RGB")
        return SourceImage(img)

    @staticmethod
    def from_numpy(arr):
        return SourceImage(Image.fromarray(arr))

    def __init__(self, img: Image, scale_factor: Union[int, float] = 1):
        self.pil_image = img
        self.binarized_cache = None
        self.array_cache = None
        self.scale_factor = scale_factor

    def scaled(self, scale_factor: float, method: RescaleMethod = RescaleMethod.NEAREST) -> 'SourceImage':
        new_size = tuple(map(lambda x: int(x*scale_factor), self.pil_image.size))

        rescaled = self.pil_image.resize(new_size, method.get_pil_scale())
        return SourceImage(rescaled, scale_factor=scale_factor)

    def scale_area(self, max_area, additional_scale_factor=None) -> 'SourceImage':

        rescale_factor = get_rescale_factor(self.pil_image, scale_area=max_area)

        if additional_scale_factor is not None:
            rescale_factor = rescale_factor * additional_scale_factor

        return self.scaled(rescale_factor)

    def binarized(self, algorithm: BinarizationAlgorithm = BinarizationAlgorithm.ISauvola,
                  params: BinarizationParams = None):
        if self.binarized_cache is None:
            if SourceImage.fail_on_binarize:
                if len(self.array().shape) == 3 or _needs_binarization(self.array()):
                    raise AssertionError("Image should already be binarized")
                return self.array()
            else:
                self.binarized_cache = binarize(self.array(), algorithm, params) * np.float32(1)
                assert self.binarized_cache.shape[:2] == self.array().shape[:2]

        return self.binarized_cache

    def array(self):
        if self.array_cache is None:
            self.array_cache = np.array(self.pil_image).astype(np.uint8)
        return self.array_cache

    def is_rescaled(self):
        return self.scale_factor != 1

    def get_width(self):
        return int(self.array().shape[1])

    def get_unscaled_width(self):
        return round(self.get_width() / self.scale_factor)

    def get_unscaled_height(self):
        return round(self.get_height() / self.scale_factor)

    def get_height(self):
        return int(self.array().shape[0])

    def get_grayscale_array(self):
        return np.array(self.pil_image.convert("L")).astype(np.uint8)

