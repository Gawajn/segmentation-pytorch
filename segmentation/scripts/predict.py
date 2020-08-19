import argparse
from os import path
import warnings
import glob
import os

from matplotlib import pyplot

from segmentation.postprocessing.baseline_extraction import extraxct_baselines_from_probability_map
from segmentation.postprocessing.layout_analysis import analyse
from segmentation.settings import PredictorSettings

warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import numpy as np
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
    parser.add_argument("--output_path_debug_images", type=str, default=None)
    parser.add_argument("--show_baselines", action="store_true")
    parser.add_argument("--show_layout", action="store_true")
    parser.add_argument("--output_xml", action="store_true")
    parser.add_argument("--output_xml_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    files = list(itertools.chain.from_iterable([glob.glob(x) for x in args.image_path]))
    networks = []
    for x in args.load:
        p_setting = PredictorSettings(MODEL_PATH=x)
        network = Network(p_setting)
        networks.append(network)
    ensemble = Ensemble(networks)
    for file in files:
        img = Image.open(file)  # open image

        p_map, scale_factor = ensemble(file, scale_area=args.scale_area)
        baselines = extraxct_baselines_from_probability_map(p_map)
        image = img.resize((int(scale_factor * img.size[0]), int(scale_factor * img.size[1])))
        if baselines is not None:
            img = img.convert('RGB')
            draw = ImageDraw.Draw(img)
            from segmentation.preprocessing.basic_binarizer import gauss_threshold
            from segmentation.preprocessing.util import to_grayscale

            grayscale = to_grayscale(np.array(image))
            binary = gauss_threshold(image=grayscale) / 255

            bboxs = analyse(baselines=baselines, image=binary, image2=image)
            bboxs = [x.scale(1 / scale_factor) for x in bboxs]
            if args.show_layout:
                for ind, x in enumerate(bboxs):
                    draw.line(x.bbox + [x.bbox[0]], fill=colors[ind % len(colors)], width=3)
            if baselines is not None and len(baselines) > 0:
                scale_baselines(baselines, 1 / scale_factor)

                for ind, x in enumerate(baselines):
                    t = list(itertools.chain.from_iterable(x))
                    a = t[::]
                    if args.show_baselines:
                        draw.line(a, fill=colors[ind % len(colors)], width=4)

                if args.output_path_debug_images:
                    basename = "debug_" + os.path.basename(file)
                    file_path = os.path.join(args.output_path_debug_images, basename)
                    img.save(file_path)
            if (args.show_baselines or args.show_layout) and args.debug:
                array = np.array(img)
                pyplot.imshow(array)
                pyplot.show()
            if args.output_xml and args.output_xml_path is not None:
                from segmentation.gui.xml_util import TextRegion, BaseLine, TextLine, XMLGenerator
                regions = []
                for box in bboxs:
                    text_lines = []
                    for b_line in box.baselines:
                        text_region_coord = b_line.baseline + list(reversed(
                            [(x, y - b_line.height) for x, y in b_line.baseline]))
                        text_lines.append(TextLine(coords=text_region_coord, baseline=BaseLine(b_line.baseline)))
                    regions.append(TextRegion(text_lines, coords=box.bbox))

                xml_gen = XMLGenerator(img.size[0], img.size[1], os.path.basename(file), regions=regions)
                print(xml_gen.baselines_to_xml_string())
                xml_gen.save_textregions_as_xml(args.output_xml_path)
                pass


if __name__ == "__main__":
    main()
