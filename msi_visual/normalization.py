
import numpy as np

def spatial_total_ion_count(img):
    processed = img / (1e-6 + np.sum(img, axis=-1)[:, :, None])
    processed = processed / (1e-6 + np.percentile(processed, 99, axis=(0, 1))[None, None, :])
    processed[processed > 1 ] = 1
    return img

def total_ion_count(img):
    return img / (1e-6 + np.sum(img, axis=-1)[:, :, None])

def median_ion(img):
    return img / (1e-6 + np.median(img, axis=-1)[:, :, None])
