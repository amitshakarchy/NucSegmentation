
import skimage.io
import numpy as np
from utils import get_tracks
import matplotlib.pyplot as plt

def single_nuc_crop(x, y, win_size, spot_frame, video):
    """
    The method returns a cropped image of the given nuclei's information
    :param x: nuclei's center x coordination
    :param y: nuclei's center y coordination
    :param win_size: diameter of the cropped image (image size is win_size X win_size)
    :param spot_frame: frame number to crop an image from
    :param video: nuclei channel video
    :return: single_nuc_crop: np.array of shape (win_size, win_size, 1) of the cropped image
    """
    single_nuc_crop = np.zeros((win_size, win_size), )
    video_tif = skimage.io.imread(video)
    video_np = np.array(video_tif)

    # Crop the needed nuclear
    single_nuc_crop = video_np[spot_frame,y-int(win_size/2):y+int(win_size/2), x-int(win_size/2):x+int(win_size/2)]

    return single_nuc_crop

