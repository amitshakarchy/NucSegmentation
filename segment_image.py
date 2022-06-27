import numpy as np
from deepcell.applications import NuclearSegmentation
import skimage.io
from utils import get_tracks
import matplotlib.pyplot as plt
from image_cropper import single_nuc_crop
from skimage.io import imread


def preprocess_image(img):
    """
    Preprocessing of a single image
    :param img: input image
    :return: reshaped image (if more preprocessing is needed, add it here!)
    """

    im = np.expand_dims(img, axis=-1)
    im = np.expand_dims(im, axis=0)

    return im


def segment_image(img):
    """
    Segmentation of a single image
    :param img: preprocessed image
    :return: seg_image- ndArray of the segmentation
    """
    app = NuclearSegmentation()

    # create the lab
    seg_image = app.predict(img)

    return seg_image


