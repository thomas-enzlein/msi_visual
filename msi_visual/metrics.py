import scipy
from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.stats import entropy
import random
import time
import sys
import numpy as np
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def get_correlation_plot(img, visualization_rgb, num_samples=300, title=None):
    visualization = cv2.cvtColor(visualization_rgb, cv2.COLOR_RGB2LAB)
    #visualization_rgb = visualization
    normalized = total_ion_count(img)
    nrow, ncol = normalized.shape[0], normalized.shape[1]
    points = np.mgrid[:nrow,:ncol].reshape(2, -1).T
    indices = list(points)

    t0 = time.time()
    
    indices = [(a, b) for a, b in indices if img[a, b, :].max() > 0]


    data = {}
    distances = ["L-2 norm", "Cosine distance", "L-∞ norm"]
    for distance in distances:
        data[distance] = {"inputs": [], "outputs": []}
    for _ in range(min(len(indices), num_samples)):
        point_a = random.choice(indices)
        point_b = random.choice(indices)
        a, b = normalized[point_a[0], point_a[1]], normalized[point_b[0], point_b[1]]
        data["L-2 norm"]['inputs'].append(np.linalg.norm(a-b, 2))
        data['Cosine distance']['inputs'].append(1 - cosine_similarity(np.array([a]), np.array([b]))[0, 0])
        data['L-∞ norm']['inputs'].append(np.abs(a-b).max())
        a, b = visualization[point_a[0], point_a[1]], visualization[point_b[0], point_b[1]]
        a = np.float32(a)
        b = np.float32(b)

        data["L-2 norm"]['outputs'].append(np.linalg.norm(a-b, 2))
        data["Cosine distance"]['outputs'].append(np.linalg.norm(a-b, 2))
        data["L-∞ norm"]['outputs'].append(np.linalg.norm(a-b, 2))
        # data['Cosine']['outputs'].append(1 - cosine_similarity(np.array([a]), np.array([b]))[0, 0])
        # data['Maximum ION difference']['outputs'].append(np.abs(a-b).max())

    fig, ax = plt.subplots()
    
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
            #corr = scipy.stats.kendalltau(inputs[relevant_indices], outputs[relevant_indices]).statistic
            corr = scipy.stats.pearsonr(inputs[relevant_indices], outputs[relevant_indices]).statistic
            graph_distances.append(int(100 * index/len(sorted_distances)))
            graph_corrs.append(corr)

        plt.plot(graph_distances, graph_corrs, label=method)
    plt.legend()
    if title:
        plt.title('Visualization distance correlation with spectra\n' + title)
    else:
        plt.title('Visualization distance correlation with spectra')
        
    plt.xlabel('Maximum input distance percentile')
    plt.ylabel('Pearson rank correlation')
    ax.set_yticks(list(np.arange(0, np.max(corr)+0.05, 0.05)))  
    return fig

def get_correlation_scatter_plot(img, visualization_rgb, num_samples=300000, title=None):
    #visualization = cv2.cvtColor(visualization_rgb, cv2.COLOR_RGB2LAB)
    visualization = visualization_rgb
    normalized = total_ion_count(img)
    nrow, ncol = normalized.shape[0], normalized.shape[1]
    points = np.mgrid[:nrow,:ncol].reshape(2, -1).T
    indices = list(points)
    t0 = time.time()
    indices = [(a, b) for a, b in indices if img[a, b, :].max() > 0]
    data = {}
    distances = ["L-2 distance", "Cosine distance", "L-∞ distance"]
    for distance in distances:
        data[distance] = {"inputs": [], "outputs": []}
    for _ in range(min(len(indices), num_samples)):
        point_a = random.choice(indices)
        point_b = random.choice(indices)
        a, b = normalized[point_a[0], point_a[1]], normalized[point_b[0], point_b[1]]
        data["L-2 distance"]['inputs'].append(np.linalg.norm(a-b, 2))
        data['Cosine distance']['inputs'].append(1 - cosine_similarity(np.array([a]), np.array([b]))[0, 0])
        data['L-∞ distance']['inputs'].append(np.abs(a-b).max())
        a, b = visualization[point_a[0], point_a[1]], visualization[point_b[0], point_b[1]]
        a = np.float32(a)
        b = np.float32(b)

        data["L-2 distance"]['outputs'].append(np.linalg.norm(a-b, 2))
        data["Cosine distance"]['outputs'].append(np.linalg.norm(a-b, 2))
        data["L-∞ distance"]['outputs'].append(np.linalg.norm(a-b, 2))

    
    outputs = np.float32(data["Cosine distance"]["outputs"])
    cosine = np.float32(data["Cosine distance"]['inputs'])
    rare = np.float32(data["L-∞ distance"]['inputs'])
    
    fig = plt.figure()
    ax = fig.add_subplot()    
    max_rank = np.maximum(cosine.argsort().argsort(), rare.argsort().argsort())
    corr = scipy.stats.pearsonr(max_rank, outputs).statistic
    expected_rank = max_rank
    result_rank = outputs.argsort().argsort()
    diff_rank = np.abs(result_rank - expected_rank)
    cm = matplotlib.pyplot.get_cmap("RdYlGn_r")
    colors = cm(np.linspace(0, 1, len(outputs)))    
    N = len(diff_rank) // 2
    cs = [colors[diff_rank[i]] for i in range(len(outputs))] #could be done with numpy's repmat
    matplotlib.pyplot.scatter(cosine,rare,color=cs,s=10, alpha=0.5)
    sm = plt.cm.ScalarMappable(cmap=cm)
    sm.set_clim(vmin=0, vmax=1)
    plt.colorbar(sm, ax=plt.gca())
    ax.set_xlabel('m/z values Cosine distance')
    ax.set_ylabel('m/z values L-∞ distance')
    if title:
        ax.set_title(title)
        
    plt.plot([], [], ' ', label=f"Max Rank Pearson Correlation {corr:.3f}")
    diff_rank_mean = np.mean(diff_rank) / len(diff_rank)
    diff_rank_mean = 100*diff_rank_mean
    plt.plot([], [], ' ', label=f"Rank deviation mean {diff_rank_mean:.3f}%")
    above_half = np.mean(diff_rank > len(diff_rank) // 2)
    above_half = 100*above_half
    plt.plot([], [], ' ', label=f"Rank deviation>50%: {above_half:.3f}%")
    plt.legend()
    return fig


if __name__ == "__main__":
    random.seed(0)
    img = np.load(sys.argv[1])
    visualization = np.array(Image.open(sys.argv[2]))
    normalized = img
    #normalized = spatial_total_ion_count(img)
    fig = get_correlation_scatter_plot(normalized, visualization, title=sys.argv[4])
    plt.savefig(sys.argv[3])
    #plt.show()