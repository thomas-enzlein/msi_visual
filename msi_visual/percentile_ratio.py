import numpy as np
import cv2

from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion

def percentile_ratio_rgb(img,
                         normalization='tic',
                         percentiles = [99.99, 99.9, 99.9, 99, 98, 85],
                         equalize=False):
    norm_funtion = {'tic': total_ion_count, 'median': median_ion, 'spatial_tic': spatial_total_ion_count}[normalization]
    sorted_normalized = norm_funtion(img)
    sorted_normalized.sort(axis=-1)
    N = sorted_normalized.shape[-1]


    p0 = sorted_normalized[:, :, int(N*percentiles[0]/100)]
    p1 = sorted_normalized[:, :, int(N*percentiles[1]/100)]
    p2 = sorted_normalized[:, :, int(N*percentiles[2]/100)]
    p3 = sorted_normalized[:, :, int(N*percentiles[3]/100)]
    p4 = sorted_normalized[:, :, int(N*percentiles[4]/100)]
    p5 = sorted_normalized[:, :, int(N*percentiles[5]/100)]


    a = p0 / (1e-5 + p1)
    a = a / np.percentile(a, 99.9)
    a[a > 1] = 1

    b = p2 / (1e-5 + p3)
    b = b / np.percentile(b, 99.9)
    b[b > 1] = 1

    c = p4 / (1e-5 + p5)
    c = c / np.percentile(c, 99.9)
    c[c > 1] = 1

    if equalize:
        visualization = cv2.merge([cv2.equalizeHist(np.uint8(255*a)),
                                   cv2.equalizeHist(np.uint8(255*b)),
                                   cv2.equalizeHist(np.uint8(255*c))])


    else:
        visualization = cv2.merge([(np.uint8(255*a)),
                                   (np.uint8(255*b)),
                                   (np.uint8(255*c))])



    visualization = cv2.cvtColor(visualization, cv2.COLOR_LAB2LRGB)

    # if equalize:
    #     visualization = cv2.merge([cv2.equalizeHist(visualization[:, :, 0]),
    #                                cv2.equalizeHist(visualization[:, :, 1]),
    #                                cv2.equalizeHist(visualization[:, :, 2])])



    visualization[img.max(axis=-1) == 0] = 0
    return visualization