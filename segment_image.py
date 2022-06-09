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
    processes_image = im

    return processes_image


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
    excel_PWD = "C:\\Users\\reut mealem\\Desktop\\FeatureAndTagTable-vertices.csv"
    video_PWD = "C:\\Users\\reut mealem\\Desktop\\211212erki-p38-stiching_s2_tdTom_ORG.tif"
    cells_table, unrelevent = get_tracks(excel_PWD)
    df2 = cells_table[['Spot position X (µm)', 'Spot position Y (µm)', 'Spot frame']]

    x = df2.iat[2, 0]
    y = df2.iat[2, 1]
    spot_frame = int(df2.iat[2, 2])
    #convert from coordinates to location in the image

    # x = int(x/0.462)
    # y = int(y/0.462)
    #
    # abc = single_nuc_crop(x,y,50,spot_frame,video_PWD)
    # plt.imshow(abc)
    # plt.show()
    #
    # plt.imsave("C:\\Users\\reut mealem\\PycharmProjects\\NucSegmentation\\resources\\manymany.png", abc)
    # skimage.io.imsave("C:\\Users\\reut mealem\\PycharmProjects\\NucSegmentation\\resources\\many.tif", abc,
    #                   check_contrast=False)
    img = imread('resources/manymany.png', as_gray=True)
    # img = imread('resources/many.tif', as_gray=True)

    preProcess = preprocess_image(img)
    seg = segment_image(preProcess)
    plt.imshow(seg.reshape(50,50,1))
    plt.show()
