from PIL import Image
from sklearn.decomposition import NMF, non_negative_factorization
import cv2
import numpy as np
from matplotlib import pyplot as plt
import joblib
from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion
from msi_visual.visualizations import visualizations_from_explanations

def normalize_channel(channel):
    channel = channel / np.percentile(channel, 99)
    channel[channel > 1] = 1
    return channel

class NMF3D:
    def __init__(self, normalization='spatial_tic', start_bin=0, end_bin=None, max_iter=200):
        self.k = 3
        self.normalization = {'tic': total_ion_count, 'median': median_ion, 'spatial_tic': spatial_total_ion_count}[normalization]
        self.start_bin = start_bin
        self.end_bin = end_bin
        self.max_iter = max_iter
        
    def fit(self, images):
        if self.end_bin is None:
            self.end_bin = images[0].shape[-1]

        vector = np.concatenate([self.normalization(img[:, :, self.start_bin:self.end_bin]).reshape(-1, images[0][:, :, self.start_bin:self.end_bin].shape[-1]) for img in images], axis=0)
        vector = vector.reshape((-1, vector.shape[-1]))
        self.model = NMF(n_components=self.k, init='random', random_state=0, max_iter=self.max_iter)
        self.W = self.model.fit_transform(vector)
        self.H = self.model.components_
    
    def get_colors(self, color_scheme='gist_rainbow'):
        _cmap = plt.cm.get_cmap(color_scheme)
        return [
            np.array(
                _cmap(i)) for i in np.arange(
                0,
                1,
                1.0 /
                self.k)]


    def visualize_training_components(self, images):
        result = []
        elements = 0
        for index, img in enumerate(images):
            img_elements = img.shape[0] * img.shape[1]
            w = self.W[elements : elements + img_elements, :].copy()
            elements = elements + img_elements
            explanations = w.transpose().reshape(self.k, img.shape[0], img.shape[1])
            explanations = explanations.transpose((1, 2, 0))

            explanations[:, :, 0] = normalize_channel(explanations[:, :, 0])
            explanations[:, :, 1] = normalize_channel(explanations[:, :, 1])
            explanations[:, :, 2] = normalize_channel(explanations[:, :, 2])
            result.append(explanations)
        return result
    
    def factorize(self, img):
        img = img[:, :, self.start_bin:self.end_bin]
        vector = self.normalization(img).reshape((-1, img.shape[-1]))
        w_new, h_new, n_iter = non_negative_factorization(vector, H=self.H, W=None, n_components=self.k, update_H=False, random_state=0)
        contributions = w_new.transpose().reshape(self.k, img.shape[0], img.shape[1])

        result = contributions.transpose((1, 2, 0))
        result[:, :, 0] = normalize_channel(result[:, :, 0])
        result[:, :, 1] = normalize_channel(result[:, :, 1])
        result[:, :, 2] = normalize_channel(result[:, :, 2])

        return result

    def visualize_factorization(self,
                                img,
                                contributions,
                                method='spatial_norm'):

        number_of_bins_for_comparison = 5
        bins = np.linspace(0, 1, number_of_bins_for_comparison)
        
        digitized_a = np.digitize(contributions[:, :, 0], bins)
        digitized_b = np.digitize(contributions[:, :, 1], bins)
        digitized_c = np.digitize(contributions[:, :, 2], bins)

        digitized = digitized_a * number_of_bins_for_comparison * number_of_bins_for_comparison + digitized_b * number_of_bins_for_comparison + digitized_c


        return digitized, np.uint8(255 * contributions)
    
    def save(self, path):
        joblib.dump(self, path)