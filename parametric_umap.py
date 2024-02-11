import numpy as np

def normalize_image_grayscale(grayscale, low_percentile: int = 2, high_percentile: int = 98):
    a = grayscale.copy()
    low = np.percentile(a[:], low_percentile)
    a = (a - low) 
    a[a < 0] = 0
    high = np.percentile(a[:], high_percentile)
    a = a / high
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