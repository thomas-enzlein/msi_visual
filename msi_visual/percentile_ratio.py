import numpy as np
import cv2

from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion

def percentile_ratio_rgb(img, normalization='tic'):
    norm_funtion = {'tic': total_ion_count, 'median': median_ion, 'spatial_tic': spatial_total_ion_count}[normalization]
    normalized = norm_funtion(img)

    sorted_normalized = np.sort(normalized, axis=-1)
    N = sorted_normalized.shape[-1]

    p_9999 = sorted_normalized[:, :, int(N*99.99/100)]
    p_999 = sorted_normalized[:, :, int(N*99.9/100)]
    p_99 = sorted_normalized[:, :, int(N*99/100)]
    p_98 = sorted_normalized[:, :, int(N*98/100)]
    p_85 = sorted_normalized[:, :, int(N*85/100)]


    a = p_9999 / (1e-5 + p_999)
    a = a / np.percentile(a, 99.9)
    a[a > 1] = 1

    b = p_999 / (1e-5 + p_99)
    b = b / np.percentile(b, 99.9)
    b[b > 1] = 1

    c = p_98 / (1e-5 + p_85)
    c = c / np.percentile(c, 99.9)
    c[c > 1] = 1

    visualization = cv2.merge([(np.uint8(255*a)), (np.uint8(255*b)), (np.uint8(255*c))])

    visualization = cv2.cvtColor(visualization, cv2.COLOR_LAB2LRGB)
    visualization[img.max(axis=-1) == 0] = 0
    return visualization