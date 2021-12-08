import os
from PIL import Image
import numpy as np
from metrics import getMetrics
from skimage.exposure import match_histograms
import pathlib
import cv2
import imageio



if __name__ == '__main__':

    path_to_domain_A='/real_A' # path for images of Domain A
    path_to_domain_B='/real_B' # path for images of Domain B

    # Calculate metrics, we pass paths as inputs
    l_feat, FID, GCF = getMetrics(path_to_norm_real_A+'/', path_to_norm_real_B+'/')
    print( l_feat, FID, GCF)

