from typing import List, Tuple

import PIL
import PIL.Image

import dataclasses_json
from segmentation.network import NetworkBase, NetworkPredictor, NewImageReconstructor, PredictionResult
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


class NetworkBaselinePredictor:
    def __init__(self, network: NetworkBase, model_config: ModelConfiguration, overwrite_color_map: ColorMap = None):
        self.predictor = NetworkPredictor(network, model_config.preprocessing_settings)
        self.model_config = model_config
        if overwrite_color_map:
            self.color_map = overwrite_color_map
        else:
            self.color_map = self.model_config.color_map

    def predict_image(self, img: SourceImage, keep_dim: bool = True, processes: int = 1) -> PIL.Image:
        res = self.predictor.predict_image(img)
        baselines = extract_baselines_from_probability_map(res.probability_map, processes=processes)

        if keep_dim:
            return BaselineResult(res,baselines)
        else:
            scale_factor = 1 / res.preprocessed_image.scale_factor
            baselines = [scale_baseline(bl, scale_factor) for bl in baselines]
            return BaselineResult(res,baselines)
