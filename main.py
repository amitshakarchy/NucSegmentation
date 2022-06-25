from skimage.io import imread
import numpy as np
from deepcell.applications import NuclearSegmentation
import matplotlib.pyplot as plt
from image_cropper import single_nuc_crop
from segment_image import preprocess_image, segment_image
from utils import get_tracks
from feature_calculator import NucleiFeatureCalculator


# read image from resources and do segmentation on the image
# the min window size that the segmentation word is 40
# excel_PWD = "C:\\Users\\reut mealem\\Desktop\\FeatureAndTagTable-vertices.csv"
# video_PWD = "C:\\Users\\reut mealem\\Desktop\\211212erki-p38-stiching_s2_tdTom_ORG_1.tif"
# cells_table, _ = get_tracks(excel_PWD)
# df2 = cells_table[['Spot position X (µm)', 'Spot position Y (µm)', 'Spot frame']]
#
# x = df2.iat[25, 0]
# y = df2.iat[25, 1]
# spot_frame = int(df2.iat[25, 2])
# #convert from coordinates to location in the image
#
# x = int(x/0.462)
# y = int(y/0.462)
#
# abc = single_nuc_crop(x, y, 40, spot_frame, video_PWD)
# plt.imshow(abc)
# plt.show()
#
# plt.imsave("C:\\Users\\reut mealem\\PycharmProjects\\NucSegmentation\\resources\\crop_5.png", abc)
# Load the image
img = imread('resources/crop_2.png', as_gray=True)
crop_size_0, crop_size_1 = img.shape[0], img.shape[1]

preProcess = preprocess_image(img)
seg = segment_image(preProcess)

seg_reshape = seg.reshape(crop_size_0, crop_size_1)
img_reshape = img.reshape(crop_size_0, crop_size_1)

# plt.imsave("C:\\Users\\reut mealem\\PycharmProjects\\NucSegmentation\\resources\\seg_2.png", seg_reshape)
# plot the original image and the predicted segmentation
fig, axs = plt.subplots(2)
axs[0].imshow(img_reshape)
axs[1].imshow(seg_reshape)
plt.show()

nuc_size = NucleiFeatureCalculator.size(seg_reshape)
print(nuc_size)

Min_Nuc, Max_Nuc, int_mean, Std_Nuc, int_sum = NucleiFeatureCalculator.intensity(seg_reshape, img_reshape)
print(Min_Nuc, Max_Nuc, int_mean, Std_Nuc, int_sum)

# size = NucleiFeatureCalculator.aspect_ratio(seg_reshape)


