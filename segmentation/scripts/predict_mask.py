import argparse
from os import path
import warnings
import glob
import os

import scipy
from skimage.filters import try_all_threshold, threshold_local
from PIL import Image, ImageDraw
from segmentation.postprocessing.baseline_extraction import extract_baselines_from_probability_map
from segmentation.postprocessing.dewarp import Dewarper
from segmentation.postprocessing.layout_analysis import analyse, connect_bounding_box, get_top_of_baselines
from segmentation.settings import PredictorSettings
from segmentation.util import PerformanceCounter

warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import numpy as np
from segmentation.util import logger


def dir_path(string):
    if path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def scale_baselines(baselines, scale_factor=1.0):
    if baselines is not None:
        for b_idx, bline in enumerate(baselines):
            for c_idx, coord in enumerate(bline):
                coord = (int(coord[0] * scale_factor), int(coord[1] * scale_factor))
                baselines[b_idx][c_idx] = coord


class Ensemble:
    def __init__(self, models):
        self.models = models

    def __call__(self, x, scale_area, additional_scale_factor=None):
        res = []
        scale_factor = None
        for m in self.models:
            p_map, s_factor = m.predict_single_image_by_path(x, rgb=True, preprocessing=True, scale_area=scale_area,
                                                             additional_scale_factor=additional_scale_factor)
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
    parser.add_argument("--scale_area", type=int, default=10000000,
                        help="max pixel amount of an image")
    parser.add_argument("--output_path_debug_images", type=str, default=None, help="Directory of the debug images")
    parser.add_argument("--cpu", action="store_true", help="Use cpu")
    parser.add_argument("--tta", action="store_true", help="Use predefined Tta-pipeline")
    parser.add_argument("--dewarp", action="store_true", help="Dewarp image using the detected baselines")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--processes", type=int, default=8)

    args = parser.parse_args()
    files = list(itertools.chain.from_iterable([glob.glob(x) for x in args.image_path]))
    networks = []
    bboxs = None
    for x in args.load:
        tta = None
        if args.tta:
            from segmentation.settings import transforms
            tta = transforms
        p_setting = PredictorSettings(MODEL_PATH=x, CPU=args.cpu, tta=tta)
        network = Network(p_setting)
        networks.append(network)
    ensemble = Ensemble(networks)
    for file in files:
        logger.info("Processing: {} \n".format(file))
        img = Image.open(file)  # open image
        scale_factor_multiplier = 1
        from matplotlib import pyplot as plt
        while True:
            p_map, scale_factor = ensemble(file, scale_area=args.scale_area,
                                           additional_scale_factor=scale_factor_multiplier)
            from matplotlib import pyplot as plt
            image2 = np.argmax(p_map, axis=-1)
            #plt.imshow(image2)
            #plt.show()
            image = img.resize((int(scale_factor * img.size[0]), int(scale_factor * img.size[1])))
            img = image.convert('RGB')
            draw = ImageDraw.Draw(img)
            #from matplotlib import pyplot as plt
            f, ax = plt.subplots(1, 2, True, True)
            ax[0].imshow(img)
            #map = scipy.special.softmax(p_map, axis=-1)
            #ax[1].imshow(img)
            ax[1].imshow(image2)

            plt.show()
            if args.output_path_debug_images:
                basename = "debug_" + os.path.basename(file)
                file_path = os.path.join(args.output_path_debug_images, basename)
                img.save(file_path)

            if args.debug:
                from matplotlib import pyplot


                from matplotlib import pyplot
                array1 = np.array(img)
                pyplot.imshow(array1)
                pyplot.show()
            break
            pass


if __name__ == "__main__":
    main()
