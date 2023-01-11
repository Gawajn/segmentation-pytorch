from typing import List, Tuple

import PIL
import PIL.Image

import dataclasses_json
from segmentation.network import NetworkBase, NetworkPredictor, NewImageReconstructor, PredictionResult, \
    NetworkPredictorBase
from segmentation.postprocessing.baseline_extraction import extract_baselines_from_probability_map
from segmentation.preprocessing.source_image import SourceImage
from segmentation.settings import ModelConfiguration, ColorMap

import numpy as np
from dataclasses import dataclass


@dataclass
class BaselineResult:
    prediction_result: PredictionResult
    base_lines: List[List[Tuple[int, int]]]


def scale_baseline(baseline, scale_factor: float = 1):
    if scale_factor == 1 or scale_factor == 1.0:
        return baseline

    return [(int(c[0] * scale_factor), int(c[1] * scale_factor)) for c in baseline]


class NetworkBaselinePostProcessor:
    @classmethod
    def from_single_predictor(cls, predictor: NetworkPredictor, mc: ModelConfiguration):
        return cls(predictor, mc.color_map)

    def __init__(self, predictor: NetworkPredictorBase, color_map: ColorMap = None):
        self.predictor = predictor
        self.color_map = color_map

    def predict_image(self, img: SourceImage, keep_dim: bool = True, processes: int = 1) -> PIL.Image:
        res = self.predictor.predict_image(img)
        baselines = extract_baselines_from_probability_map(res.probability_map, processes=processes)

        if keep_dim:
            scale_factor = 1 / res.preprocessed_image.scale_factor
            baselines = [scale_baseline(bl, scale_factor) for bl in baselines] if baselines else []
            return BaselineResult(res, baselines)
        else:
            return BaselineResult(res, baselines)


@dataclass
class MaskPredictionResult:
    prediction_result: PredictionResult
    generated_mask: PIL.Image


class NetworkMaskPostProcessor:
    @classmethod
    def from_single_predictor(cls, predictor: NetworkPredictor, mc: ModelConfiguration):
        return cls(predictor, mc.color_map)

    def __init__(self, predictor: NetworkPredictorBase, color_map: ColorMap = None):
        self.predictor = predictor
        self.color_map = color_map

    def predict_image(self, img: SourceImage, keep_dim: bool = True) -> PIL.Image:
        res = self.predictor.predict_image(img)

        # create labeled image from probability map
        lmap = np.argmax(res.probability_map, axis=-1)
        mask = NewImageReconstructor.label_to_colors(lmap, self.color_map)

        outimg = PIL.Image.fromarray(mask, mode="RGB")

        if keep_dim:
            mask = outimg.resize(size=(img.get_width(), img.get_height()), resample=PIL.Image.NEAREST)
        else:
            mask = outimg

        mpr = MaskPredictionResult(prediction_result=res, generated_mask=mask)
        return mpr
