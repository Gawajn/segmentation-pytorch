from enum import Enum, auto
import doxapy
import skimage.color
import numpy as np

import segmentation.binarization.ocrupus as ocropus


class ParameterUnavailable(Exception):
    pass


class InvalidAlgorithm(Exception):
    pass


class BinarizationAlgorithm(Enum):
    Threshold = "threshold"
    Ocropus = "ocropus"
    # doxa methods
    ISauvola = "isauvola"
    Sauvola = "sauvola"
    Gatos = "gatos"
    Otsu = "otsu"
    Wolf = "wolf"
    Bernsen = "bernsen"
    Wan = "wan"

    @classmethod
    def doxa_methods(cls):
        return {cls.ISauvola, cls.Sauvola, cls.Otsu, cls.Wan,
                cls.Wolf, cls.Bernsen, cls.Gatos}


def available_params(alg: BinarizationAlgorithm):
    return set(["window", "k"])


class BinarizationParams:
    def __init__(self, alg: BinarizationAlgorithm, params: dict = dict()):
        avail_params = available_params(alg)
        for param in params.keys():
            if param not in avail_params:
                raise ParameterUnavailable(f"Parameter {param} unavailable for {alg.value}")
        self.params = params


def _binarize_doxapy(img, alg: doxapy.Binarization.Algorithms, params: BinarizationParams):
    bins = doxapy.Binarization(alg)
    bins.initialize(img)
    binarized = np.empty(img.shape, dtype=np.uint8)
    bins.to_binary(binarized, params.params)

    return binarized > 0


def _binarize_to_grey(image):
    if len(image.shape) == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            # convert the image to greyscale
            # important to use this color space!!!!
            image = skimage.color.rgb2gray(image)
            image = (image * 255).astype(np.uint8)
    # check if the image is already binarized
    if not image.dtype == np.uint8:
        image = np.array(image, dtype=np.uint8)
    return image


def _needs_binarization(grey):
    return not np.all(np.logical_or(grey == 0, grey == 255))


def _binarize_threshold(grey, threshold=127):
    return grey >= threshold


def binarize(image: np.ndarray, algorithm: BinarizationAlgorithm = BinarizationAlgorithm.ISauvola,
             params: BinarizationParams = None):
    """Binarize an image stored in a numpy array

        Parameters:
        image(np.ndarray): The Image to binarize. Will automatically be converted to greyscale
        algorithm(BinarizationAlgorithm): Algorithm to use for the bianrization (defaults to ISauvola)
        params(BinarizationParams): Parameters for the given binarizier(defaults to default params for the given binarizer)

        Returns:
        Image as a np.ndarray of type bool, with 0 == Foreground (black) and 1==Background(white)
       """
    if params is None:
        params = BinarizationParams(algorithm, dict())

    grey = _binarize_to_grey(image)
    if not _needs_binarization(grey):
        return grey > 0

    if algorithm in BinarizationAlgorithm.doxa_methods():
        alg = None
        if algorithm == BinarizationAlgorithm.ISauvola:
            alg = doxapy.Binarization.Algorithms.ISAUVOLA
        elif algorithm == BinarizationAlgorithm.Sauvola:
            alg = doxapy.Binarization.Algorithms.SAUVOLA
        elif algorithm == BinarizationAlgorithm.Otsu:
            alg = doxapy.Binarization.Algorithms.OTSU
        elif algorithm == BinarizationAlgorithm.Gatos:
            alg = doxapy.Binarization.Algorithms.GATOS
        elif algorithm == BinarizationAlgorithm.Wolf:
            alg = doxapy.Binarization.Algorithms.WOLF
        elif algorithm == BinarizationAlgorithm.Bernsen:
            alg = doxapy.Binarization.Algorithms.BERNSEN
        elif algorithm == BinarizationAlgorithm.Wan:
            alg = doxapy.Binarization.Algorithms.WAN
        else:
            raise InvalidAlgorithm(f"Algorithm {algorithm} is not available")
        return _binarize_doxapy(grey, alg, params)
    elif algorithm == BinarizationAlgorithm.Ocropus:
        return ocropus.binarize(grey / 255.0, assert_float_greyscale=True)
    elif algorithm == BinarizationAlgorithm.Threshold:
        return _binarize_threshold(grey)
    else:
        raise InvalidAlgorithm(f"Algorithm {algorithm} is not available")
