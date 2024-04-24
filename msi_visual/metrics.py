import scipy
from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.stats import entropy
import random
import time
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def get_correlation_plot(img, visualization_rgb, num_samples=500, title=None):
    visualization = cv2.cvtColor(visualization_rgb, cv2.COLOR_RGB2LAB)
    #visualization_rgb = visualization
    normalized = total_ion_count(img)
    nrow, ncol = normalized.shape[0], normalized.shape[1]
    points = np.mgrid[:nrow,:ncol].reshape(2, -1).T
    indices = list(points)

    t0 = time.time()
    
    indices = [(a, b) for a, b in indices if img[a, b, :].max() > 0]


    data = {}
    distances = ["Euclidean", "Cosine", "Maximum ION difference"]
    for distance in distances:
        data[distance] = {"inputs": [], "outputs": []}
    for _ in range(min(len(indices), num_samples)):
        point_a = random.choice(indices)
        point_b = random.choice(indices)
        a, b = normalized[point_a[0], point_a[1]], normalized[point_b[0], point_b[1]]
        data["Euclidean"]['inputs'].append(np.linalg.norm(a-b, 2))
        data['Cosine']['inputs'].append(1 - cosine_similarity(np.array([a]), np.array([b]))[0, 0])
        data['Maximum ION difference']['inputs'].append(np.abs(a-b).max())
        a, b = visualization[point_a[0], point_a[1]], visualization[point_b[0], point_b[1]]
        a = np.float32(a)
        b = np.float32(b)

        data["Euclidean"]['outputs'].append(np.linalg.norm(a-b, 2))
        data["Cosine"]['outputs'].append(np.linalg.norm(a-b, 2))
        data["Maximum ION difference"]['outputs'].append(np.linalg.norm(a-b, 2))
        # data['Cosine']['outputs'].append(1 - cosine_similarity(np.array([a]), np.array([b]))[0, 0])
        # data['Maximum ION difference']['outputs'].append(np.abs(a-b).max())

    fig = plt.figure()

    for method in data:
        inputs = np.float32(data[method]["inputs"])
        outputs = np.float32(data[method]["outputs"])
        sorted_distances = sorted(inputs)
        indices = list(range(len(sorted_distances)//10, len(sorted_distances), len(sorted_distances)//10))
        indices.append(len(sorted_distances)-1)
        graph_distances, graph_corrs = [], []
        for index in indices:
            d = sorted_distances[index]
            relevant_indices = [i for i in range(len(inputs)) if inputs[i] < d]
            corr = scipy.stats.kendalltau(inputs[relevant_indices], outputs[relevant_indices]).statistic
            graph_distances.append(int(100 * index/len(sorted_distances)))
            graph_corrs.append(corr)

        plt.plot(graph_distances, graph_corrs, label=method)
    plt.legend()
    if title:
        plt.title('Visualization distance correlation with spectra\n' + title)
    else:
        plt.title('Visualization distance correlation with spectra')
        
    plt.xlabel('Maximum input distance percentile')
    plt.ylabel('Kendall rank correlation')
    return fig

if __name__ == "__main__":
    random.seed(0)
    img = np.load(sys.argv[1])
    visualization = np.array(Image.open(sys.argv[2]))
    normalized = img
    #normalized = spatial_total_ion_count(img)
    fig = get_correlation_plot(normalized, visualization, title=sys.argv[4])
    plt.savefig(sys.argv[3])
    plt.show()