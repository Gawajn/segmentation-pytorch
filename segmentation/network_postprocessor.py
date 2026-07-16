from typing import List, Optional, Tuple

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


@dataclass
class BaseLinePostProcessorConfig:
    min_cc_area = 10
    max_cc_distance = 100


def scale_baseline(baseline, scale_factor: float = 1):
    if scale_factor == 1 or scale_factor == 1.0:
        return baseline

    return [(int(c[0] * scale_factor), int(c[1] * scale_factor)) for c in baseline]


class NetworkBaselinePostProcessor:
    @classmethod
    def from_single_predictor(cls, predictor: NetworkPredictor, mc: ModelConfiguration):
        return cls(predictor, mc.color_map)

    def __init__(self, predictor: NetworkPredictorBase, color_map: ColorMap = None,
                 base_line_post_processor_config=BaseLinePostProcessorConfig()):
        self.predictor = predictor
        self.color_map = color_map
        self.config = base_line_post_processor_config

    def predict_image(self, img: SourceImage, keep_dim: bool = True, processes: int = 1) -> PIL.Image:
        res = self.predictor.predict_image(img)
        baselines = extract_baselines_from_probability_map(res.probability_map, processes=processes,
                                                           min_cc_area=self.config.min_cc_area,
                                                           max_cc_distance=self.config.max_cc_distance)

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
    additional_generated_masks: Optional[List[PIL.Image.Image]] = None


class NetworkMaskPostProcessor:
    @classmethod
    def from_single_predictor(cls, predictor: NetworkPredictor, mc: ModelConfiguration):
        return cls(predictor, mc.color_map, additional_color_maps=getattr(mc, "additional_color_maps", None))

    def __init__(self, predictor: NetworkPredictorBase, color_map: ColorMap = None,
                 additional_color_maps: Optional[List[ColorMap]] = None):
        self.predictor = predictor
        self.color_map = color_map
        self.additional_color_maps = additional_color_maps

    @staticmethod
    def probability_map_to_pil(probability_map: np.ndarray, color_map: ColorMap, img: SourceImage,
                               keep_dim: bool) -> PIL.Image.Image:
        lmap = np.argmax(probability_map, axis=-1)
        mask = NewImageReconstructor.label_to_colors(lmap, color_map)

        outimg = PIL.Image.fromarray(mask, mode="RGB")

        if keep_dim:
            return outimg.resize(size=(img.get_width(), img.get_height()), resample=PIL.Image.NEAREST)
        return outimg

    def predict_image(self, img: SourceImage, keep_dim: bool = True) -> PIL.Image:
        res = self.predictor.predict_image(img)

        # create labeled image from probability map
        mask = self.probability_map_to_pil(res.probability_map, self.color_map, img, keep_dim)

        additional_masks = None
        if res.other_probability_map and self.additional_color_maps:
            additional_masks = [
                self.probability_map_to_pil(pmap, cmap, img, keep_dim)
                for pmap, cmap in zip(res.other_probability_map, self.additional_color_maps)
            ]

        mpr = MaskPredictionResult(prediction_result=res, generated_mask=mask,
                                   additional_generated_masks=additional_masks)
        return mpr
