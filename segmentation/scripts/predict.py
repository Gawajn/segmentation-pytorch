import argparse
from os import path
import warnings
import glob
import os

from PIL import Image, ImageDraw
from tqdm import tqdm

from segmentation.model_builder import ModelBuilderLoad
from segmentation.network import EnsemblePredictor
from segmentation.network_postprocessor import NetworkMaskPostProcessor, NetworkBaselinePostProcessor
from segmentation.preprocessing.source_image import SourceImage
from segmentation.scripts.train import get_default_device

warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import numpy as np


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
    colors = [(255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 0),
              (0, 255, 255),
              (255, 0, 255)]
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, nargs="*", default=[],
                        help="load models and use it for inference (specify .torch file)")
    parser.add_argument("--image_path", type=str, nargs="*", default=[], help="load models and use it for inference")
    parser.add_argument("--scale_area", action="store_true", help="Enable image scaling")
    parser.add_argument("--scale_area_size", type=int, help="max pixel amount of an image")

    parser.add_argument("--output_xml", action="store_true", help="Outputs Xml Files")
    parser.add_argument("--output_xml_path", type=str, default=None, help="Directory of the XML output")
    parser.add_argument("--mode", choices=["xml_baseline", "xml_region", "mask"], required=True)

    parser.add_argument("--max_line_height", type=int, default=None,
                        help="If the average line_height of an document is bigger then the specified value, "
                             "the document is scaled down an processed again on the new resolution. "
                             "Proposed Value == 22")
    parser.add_argument("--min_line_height", type=int, default=None,
                        help="If the average line_height of an document is smaller then the specified value, "
                             "the document is scaled up an processed again on the new resolution")
    parser.add_argument("-d", "--device", type=str, default=get_default_device())
    parser.add_argument("--tta", action="store_true", help="Use predefined Tta-pipeline")
    parser.add_argument("--dewarp", action="store_true", help="Dewarp image using the detected baselines")
    parser.add_argument("--show_result", action="store_true")
    parser.add_argument("--output_path_debug_images", type=str, default=None, help="Directory of the debug images")

    parser.add_argument("--processes", type=int, default=8)

    args = parser.parse_args()
    image_list = sorted(itertools.chain.from_iterable([glob.glob(x) for x in args.image_path]))
    base_model_files = [ModelBuilderLoad.from_disk(model_weights=i, device=args.device) for i in args.load]
    base_models = [i.get_model() for i in base_model_files]
    base_configs = [i.get_model_configuration() for i in base_model_files]
    preprocessing_settings = [i.get_model_configuration().preprocessing_settings for i in base_model_files]
    if args.scale_area:
        for i in preprocessing_settings:
            i.scale_max_area = args.scale_area_size
            i.scale_predict = args.scale_area

    predictor = EnsemblePredictor(base_models, preprocessing_settings)
    config = base_configs[0]

    if args.mode == "mask" or args.mode == "xml_region":
        nmaskpred = NetworkMaskPostProcessor(predictor, config.color_map)
        for img_path in tqdm(image_list):
            simg = SourceImage.load(img_path)
            mask = nmaskpred.predict_image(simg)
            if args.show_result:
                mask.generated_mask.show()
            if args.output_path_debug_images:
                basename = "debug_" + os.path.basename(img_path)
                file_path = os.path.join(args.output_path_debug_images, basename)
                mask.generated_mask.save(file_path)

    if args.mode == "xml_baseline":
        nbaselinepred = NetworkBaselinePostProcessor(predictor, config.color_map)
        for img_path in tqdm(image_list):
            simg = SourceImage.load(img_path)
            simg.pil_image = simg.pil_image.convert('RGB')
            draw = ImageDraw.Draw(simg.pil_image)

            result = nbaselinepred.predict_image(simg)
            for ind, x in enumerate(result.base_lines):

                t = list(itertools.chain.from_iterable(x))
                a = t[::]
                if args.show_result:
                    draw.line(a, fill=colors[ind % len(colors)], width=4)
            if args.output_path_debug_images:
                basename = "debug_" + os.path.basename(img_path)
                file_path = os.path.join(args.output_path_debug_images, basename)
                simg.pil_image.save(file_path)

            if args.output_xml and args.output_xml_path is not None:
                from segmentation.gui.xml_util import TextRegion, BaseLine, TextLine, XMLGenerator
                regions = []
                if result.base_lines is not None:
                    text_lines = []
                    for b_line in result.base_lines:
                        text_lines.append(TextLine(coords=b_line + list(reversed(b_line)), baseline=BaseLine(b_line)))
                    regions.append(TextRegion(text_lines, coords=b_line + list(reversed(b_line))))

                xml_gen = XMLGenerator(simg.get_width(),simg.get_height(), os.path.basename(img_path), regions=regions)
                xml_gen.save_textregions_as_xml(args.output_xml_path)


if __name__ == "__main__":
    main()
