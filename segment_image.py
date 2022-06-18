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

if __name__ == '__main__':

    #read image from resources and do segmentation on the image
    #the min window size that the segmentation word is 40

    img = imread('resources/crop_2.png', as_gray=True)
    crop_size_0, crop_size_1 = img.shape[0], img.shape[1]

    preProcess = preprocess_image(img)
    seg = segment_image(preProcess)

    plt.imshow(seg.reshape(crop_size_0, crop_size_1, 1))
    plt.show()
