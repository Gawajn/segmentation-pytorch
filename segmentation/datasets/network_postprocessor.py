from dataclasses import dataclass
from typing import List, Tuple

import PIL
import numpy as np

from segmentation.network import NetworkPredictor, NetworkPredictorBase, PredictionResult, NewImageReconstructor
from segmentation.network_postprocessor import scale_baseline
from segmentation.postprocessing.tables.table_extraction import extract_tables_from_probability_map
from segmentation.preprocessing.source_image import SourceImage
from segmentation.settings import ModelConfiguration, ColorMap


@dataclass
class TableResult:
    prediction_result: PredictionResult
    horizontal_lines: List[List[Tuple[int, int]]]
    vertical_lines: List[List[Tuple[int, int]]]
    mask: PIL.Image


def transpose_baselines(lines: List[List[Tuple[int, int]]]):
    transposed_lines = []
    for line in lines:
        transpose_line = []
        for point in line:
            transpose_line.append((point[1], point[0]))
        transposed_lines.append(transpose_line)
    return transposed_lines


class NetworkTablePostProcessor:
    @classmethod
    def from_single_predictor(cls, predictor: NetworkPredictor, mc: ModelConfiguration):
        return cls(predictor, mc.color_map)

    def __init__(self, predictor: NetworkPredictorBase, color_map: ColorMap = None):
        self.predictor = predictor
        self.color_map = color_map

    def predict_image(self, img: SourceImage, keep_dim: bool = True, processes: int = 1) -> PIL.Image:
        res = self.predictor.predict_image(img)
        baselines_horizontal = extract_tables_from_probability_map(res.probability_map, processes=processes, line_horizontal_index=1, line_vertical_index=3)
        baselines_vertical = transpose_baselines(
            extract_tables_from_probability_map(np.transpose(res.probability_map, axes=[1, 0, 2]),
                                                processes=processes, line_horizontal_index=2, line_vertical_index=3))
        mask = None
        if self.color_map:
            lmap = np.argmax(res.probability_map, axis=-1)
            mask = NewImageReconstructor.label_to_colors(lmap, self.color_map)

            outimg = PIL.Image.fromarray(mask, mode="RGB")

            if keep_dim:
                mask = outimg.resize(size=(img.get_width(), img.get_height()), resample=PIL.Image.NEAREST)
            else:
                mask = outimg
        if keep_dim:
            scale_factor = 1 / res.preprocessed_image.scale_factor
            baselines_horizontal = [scale_baseline(bl, scale_factor) for bl in
                                    baselines_horizontal] if baselines_horizontal else []
            baselines_vertical = [scale_baseline(bl, scale_factor) for bl in
                                  baselines_vertical] if baselines_vertical else []
            return TableResult(res, baselines_horizontal, baselines_vertical, mask)
        else:
            return TableResult(res, baselines_horizontal, baselines_vertical, mask )
