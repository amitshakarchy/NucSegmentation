from skimage.io import imread
import numpy as np
from deepcell.applications import NuclearSegmentation
import matplotlib.pyplot as plt


# Load the image
image = imread('resources/Screenshot 2022-05-12 115423.png', as_gray=True)

# Expand image dimensions to rank 4
im = np.expand_dims(image, axis=-1)
im = np.expand_dims(im, axis=0)

# Create the application
app = NuclearSegmentation()

# create the lab
labeled_image = app.predict(im)

# plot the original image and the predicted segmentation
fig, axs = plt.subplots(2)
axs[0].imshow(image.reshape(72, 102, 1))
axs[1].imshow(labeled_image.reshape(72, 102, 1))
plt.show()