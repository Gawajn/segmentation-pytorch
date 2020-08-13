import argparse
import json
from os import path
from typing import List
import warnings
import glob

from matplotlib import pyplot

from segmentation.postprocessing.baseline_extraction import extraxct_baselines_from_probability_map
from segmentation.settings import PredictorSettings
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import numpy as np
from segmentation.postprocessing import baseline_extraction
from PIL import Image, ImageDraw


def dir_path(string):
    if path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def scale_baselines(baselines, scale_factor=1.0):
    for b_idx, bline in enumerate(baselines):
        for c_idx, coord in enumerate(bline):
            coord = (int(coord[0] * scale_factor), int(coord[1] * scale_factor))
            baselines[b_idx][c_idx] = coord


class Ensemble:
    def __init__(self, models):
        self.models = models

    def __call__(self, x, scale_area):
        res = []
        scale_factor = None
        for m in self.models:
            p_map, s_factor = m.predict_single_image_by_path(x, rgb=True, preprocessing=True, scale_area=1000000)
            scale_factor = s_factor
            res.append(p_map)
        if len(res) == 1:
            return res[0], scale_factor
        res = np.stack(res, axis=0)
        return np.mean(res, axis=0), scale_factor


def main():
    from segmentation.network import TrainSettings, dirs_to_pandaframe, load_image_map_from_file, MaskSetting, MaskType, \
        PCGTSVersion, XMLDataset, Network, compose, MaskGenerator, MaskDataset
    from segmentation.settings import Architecture
    from segmentation.modules import ENCODERS
    colors = [(255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 0),
              (0, 255, 255),
              (255, 0, 255)]

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, nargs="*", default=[],
                        help="load models and use it for inference")
    parser.add_argument("--image_path", type=str, nargs="*", default=[],
                        help="load models and use it for inference")
    parser.add_argument("--scale_area", type=int, default=1000000,
                        help="max pixel amount of an image")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--show_baselines", action="store_true")

    args = parser.parse_args()
    files = list(itertools.chain.from_iterable([glob.glob(x) for x in args.image_path]))
    networks = []
    for x in args.load:
        p_setting = PredictorSettings(MODEL_PATH=x)
        network = Network(p_setting)
        networks.append(network)
    ensemble = Ensemble(networks)
    for file in files:
        p_map, scale_factor = ensemble(file, scale_area=args.scale_area)
        baselines = extraxct_baselines_from_probability_map(p_map)
        if baselines is not None and len(baselines) > 0:
            scale_baselines(baselines, 1 / scale_factor)
            img = Image.open(file)  # open image
            img = img.convert('RGB')
            draw = ImageDraw.Draw(img)

            for ind, x in enumerate(baselines):
                t = list(itertools.chain.from_iterable(x))
                a = t[::]
                draw.line(a, fill=colors[ind % len(colors)], width=4)
            if args.output_path:
                import os
                basename = "debug_" + os.path.basename(file)
                file_path = os.path.join(args.output_path, basename)
                img.save(file_path)
            if args.show_baselines:
                array = np.array(img)
                pyplot.imshow(array)
                pyplot.show()


if __name__ == "__main__":
    main()
