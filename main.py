from skimage.io import imread
import numpy as np
from deepcell.applications import NuclearSegmentation
import matplotlib.pyplot as plt
from image_cropper import single_nuc_crop
from segment_image import preprocess_image, segment_image
from utils import get_tracks


# read image from resources and do segmentation on the image
# the min window size that the segmentation word is 40

# Load the image
img = imread('resources/crop_2.png', as_gray=True)
crop_size_0, crop_size_1 = img.shape[0], img.shape[1]

preProcess = preprocess_image(img)
seg = segment_image(preProcess)

# plot the original image and the predicted segmentation
fig, axs = plt.subplots(2)
axs[0].imshow(img.reshape(crop_size_0, crop_size_1, 1))
axs[1].imshow(seg.reshape(crop_size_0, crop_size_1, 1))
plt.show()


