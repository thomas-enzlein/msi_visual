import numpy as np
from sklearn.decomposition import non_negative_factorization
from matplotlib import pyplot as plt
from collections import defaultdict

from msi_visual.visualizations import get_colors, visualizations_from_explanations

def brain_nmf_semantic_segmentation(img_path_for_segmentation: str, H_path: str = "h_cosegmentation.npy", NUM_COMPONENTS:int = 5) -> np.ndarray:
    colors_for_components = get_colors(NUM_COMPONENTS)
    img_for_segmentation = np.load(img_path_for_segmentation)[:, :, 600 : ]
    H = np.load(H_path)
    img_for_segmentation = img_for_segmentation / (1e-6 + np.median(img_for_segmentation, axis=-1)[:, :, None])
    vector = img_for_segmentation.reshape((-1, img_for_segmentation.shape[-1]))

    w_new, h_new, n_iter = non_negative_factorization(vector, H=H, W=None, n_components=NUM_COMPONENTS, update_H=False, random_state=0)
    explanations = w_new.transpose().reshape(NUM_COMPONENTS, img_for_segmentation.shape[0], img_for_segmentation.shape[1])
    explanations[4, :] = 0
    spatial_sum_visualization, global_percentile_visualization, normalized_sum, normalized_percentile = visualizations_from_explanations(img_for_segmentation, explanations, colors_for_components)
    return normalized_sum.argmax(axis=0)

def brain_nmf_semantic_segmentation_highres(img_path_for_segmentation: str, H_path: str = "h_cosegmentation_tims.npy", NUM_COMPONENTS:int = 20) -> np.ndarray:
    colors_for_components = get_colors(NUM_COMPONENTS)
    img_for_segmentation = np.load(img_path_for_segmentation)[:, :, : ]
    H = np.load(H_path)
    img_for_segmentation = img_for_segmentation / (1e-6 + np.sum(img_for_segmentation, axis=-1)[:, :, None])
    vector = img_for_segmentation.reshape((-1, img_for_segmentation.shape[-1]))

    w_new, h_new, n_iter = non_negative_factorization(vector, H=H, W=None, n_components=NUM_COMPONENTS, update_H=False, random_state=0)
    explanations = w_new.transpose().reshape(NUM_COMPONENTS, img_for_segmentation.shape[0], img_for_segmentation.shape[1])
    spatial_sum_visualization, global_percentile_visualization, normalized_sum, normalized_percentile = visualizations_from_explanations(img_for_segmentation, explanations, colors_for_components)
    return normalized_sum.argmax(axis=0)

import numpy as np

def normalize_image_grayscale(grayscale, low_percentile: int = 0.1, high_percentile: int = 99.9):
    a = grayscale.copy()
    low = np.percentile(a[:], low_percentile)
    a = (a - low) 
    a[a < 0] = 0
    high = np.percentile(a[:], high_percentile)
    a = a / (1e-7+high)
    a[a < 0] = 0
    a [ a > 1 ] = 1
    return a


def image_histogram_equalization(image, mask, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image[mask > 0].flatten(), number_bins, density=True)
    
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)