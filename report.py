from sklearn.decomposition import NMF
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import cv2
import numpy as np
from typing import List, Dict
import math
import glob
from PIL import Image
from matplotlib import pyplot as plt
import math
import tqdm
import warnings
import pathlib
import sys
import os
import joblib
import matplotlib.pyplot as plt
import umap
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

def get_umap(results, colors_for_components):
    fig = plt.figure()
    all_embeddings = []
    all_labels = []
    all_colors = []
    all_markers = []
    marker_list = ['x', 'o', "D", "<", "*", "p", "v"]
    colors = colors_for_components
    regions = [result["name"] for result in results]
    for region_index, result in enumerate(results):
        name = result["name"]
        W, H = result["components"]
        for i in range(len(H)):
            all_embeddings.append(H[i, :])
            all_labels.append(f"Region: {name} Component: {i}")
            all_colors.append(colors[i])
            marker = marker_list[region_index]
            all_markers.append(marker)
    all_embeddings = np.float32(all_embeddings)
    normalized = all_embeddings
    # normalized = all_embeddings - np.mean(all_embeddings, axis = 0)
    # normalized = normalized / (1e-5 + np.std(normalized, axis = 0))
    plt.gca().set_aspect("equal", "datalim")
    umap_embeddings = umap.UMAP(
        n_neighbors=5, min_dist=0.9, verbose=True
    ).fit_transform(normalized)
    count = len(all_labels)
    for index, marker in enumerate(set(all_markers)):
        indices = [i for i in range(len(all_markers)) if all_markers[i] == marker]
        sc = plt.scatter(*umap_embeddings[indices, :].T, s=30, c=[all_colors[i] for i in indices], 
                    marker=marker, alpha=1.0, label=f"{index}")
    plt.legend(loc='lower left')
    plt.title(f"UMAP m/z vectors for regions and components")
    plt.axis('off')
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plt.close(fig=fig)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return Image.fromarray(data)
    


def show_factorization_on_image(img: np.ndarray,
                                explanations: np.ndarray,
                                colors: List[np.ndarray] = None,
                                image_weight: float = 0.5,
                                concept_labels: List = None) -> np.ndarray:
    """ Color code the different component heatmaps on top of the image.
        Every component color code will be magnified according to the heatmap itensity
        (by modifying the V channel in the HSV color space),
        and optionally create a lagend that shows the labels.

        Since different factorization component heatmaps can overlap in principle,
        we need a strategy to decide how to deal with the overlaps.
        This keeps the component that has a higher value in it's heatmap.

    :param img: The base image RGB format.
    :param explanations: A tensor of shape num_componetns x height x width, with the component visualizations.
    :param colors: List of R, G, B colors to be used for the components.
                   If None, will use the gist_rainbow cmap as a default.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * visualization.
    :concept_labels: A list of strings for every component. If this is paseed, a legend that shows
                     the labels and their colors will be added to the image.
    :returns: The visualized image.
    """
    n_components = explanations.shape[0]
    if colors is None:
        # taken from https://github.com/edocollins/DFF/blob/master/utils.py
        _cmap = plt.cm.get_cmap('gist_rainbow')
        colors = [
            np.array(
                _cmap(i)) for i in np.arange(
                0,
                1,
                1.0 /
                n_components)]
    concept_per_pixel = explanations.argmax(axis=0)
    masks = []
    for i in range(n_components):
        mask = np.zeros(shape=(img.shape[0], img.shape[1], 3))
        mask[:, :, :] = colors[i][:3]
        explanation = explanations[i]
        explanation[concept_per_pixel != i] = 0
        mask = np.uint8(mask * 255)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)
        mask[:, :, 2] = np.uint8(255 * explanation)
        mask = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
        mask = np.float32(mask) / 255
        masks.append(mask)

    mask = np.sum(np.float32(masks), axis=0)
    result = img * image_weight + mask * (1 - image_weight)
    result = np.uint8(result * 255)

    if concept_labels is not None:
        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        fig = plt.figure(figsize=(result.shape[1] * px, result.shape[0] * px))
        plt.rcParams['legend.fontsize'] = int(
            14 * result.shape[0] / 256 / max(1, n_components / 6))
        lw = 5 * result.shape[0] / 256
        lines = [Line2D([0], [0], color=colors[i], lw=lw)
                 for i in range(n_components)]
        plt.legend(lines,
                   concept_labels,
                   mode="expand",
                   fancybox=True,
                   shadow=True)

        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.axis('off')
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt.close(fig=fig)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = cv2.resize(data, (result.shape[1], result.shape[0]))
        result = np.hstack((result, data))
    return result

def get_colors(number):
    _cmap = plt.cm.get_cmap('gist_rainbow')
    colors_for_components = [
        np.array(
            _cmap(i)) for i in np.arange(
            0,
            1,
            1.0 /
            NUM_COMPONENTS)]
    return colors_for_components

def get_unsupervised_decomposition(path: str, 
                                   NUM_COMPONENTS: int,
                                   colors: list,
                                   start_mz: int = 300):
    name = pathlib.Path(path).stem
    img = np.load(path)
    img = np.float32(img)
    img = img[:, :, 300 : ]    
    vector = img.reshape((-1, img.shape[-1]))
    model = NMF(n_components=NUM_COMPONENTS, init='random', random_state=0)
    W = model.fit_transform(vector)
    H = model.components_
    
    model = joblib.load("kmeans.joblib")
    centers = model.cluster_centers_
    order = []
    diffs = []
    for i in range(len(H)):
        sim = cosine_similarity(H[i, :][None, :], centers)[0, :]
        diffs.append(sim.max())
        order.append(sim.argmax())
    colors_ordered = [colors[i] for i in order]
    
    explanations = W.transpose().reshape(NUM_COMPONENTS, img.shape[0], img.shape[1])
    explanations = explanations / explanations.sum(axis=(1, 2))[:, None, None]
    explanations = explanations / np.percentile(explanations, 99, axis=(1, 2))[:, None, None]
    explanations = explanations[0 : , :, :]
    
    mask = np.uint8(explanations.argmax(axis=0) == 0)
    mask = mask * explanations[0, :]

    img = img / img.max(axis=(0, 1))[None, None, :]
    visualization = show_factorization_on_image(np.zeros(shape=((img.shape[0], img.shape[1], 3))),
                                                explanations,
                                                image_weight=0.0,
                                                colors=colors)
    visualization2 = show_factorization_on_image(np.zeros(shape=((img.shape[0], img.shape[1], 3))),
                                                explanations,
                                                image_weight=0.0,
                                                colors=colors_ordered)
    images = []
    for i in range(NUM_COMPONENTS):
        fig = plt.figure()
        low = np.min(H[i, :])
        high = np.max(H[i, :])
        plt.ylim([0, math.ceil(high+2)])
        plt.tight_layout()
        plt.bar(range(600, 1301), H[i, :], color = colors[i])
        plt.title(f'Region: {name} Component: {i}', y=0.9)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig=fig)
        images.append(data)
    images = np.vstack(images)

    visualization = np.hstack((visualization, visualization2))

    table = np.zeros((NUM_COMPONENTS, 3))
    table[:, 0] = [f"{o:2d}" for o in list(range(NUM_COMPONENTS))]
    table[:, 1] = [f"{o:2d}" for o in np.int32(order)]
    table[:, 2] = [f"{d:.3f}" for d in diffs]

    fig = plt.figure()
    plt.table(table, colLabels=["Component", "Reference Comp.", "Similarity"],
              loc='upper left')
    plt.title(f'Similarity to reference components\n{name}')
    fig.tight_layout()
    fig.patch.set_visible(False)
    plt.axis('off')
    plt.axis('tight')    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig=fig)
    diff_table = Image.fromarray(data)

    result = {"visualization": Image.fromarray(visualization),
              "mz_plots": Image.fromarray(images), 
              "components": (W, H),
              "path": path,
              "name": name,
              "order": order,
              "diffs": diffs,
              "diff_table": diff_table}
    return result

paths = glob.glob(os.path.join(sys.argv[1], "*.npy"))
NUM_COMPONENTS = int(sys.argv[3])
colors = get_colors(NUM_COMPONENTS)

results = []
for path in tqdm.tqdm(paths):
    result = get_unsupervised_decomposition(path,
                                            NUM_COMPONENTS, 
                                            colors=colors)
    results.append(result)

images_for_pdf = []
visualizations = [result["visualization"] for result in results]
mz_plots = [result["mz_plots"] for result in results]
diff_tables = [result["diff_table"] for result in results]

umap_plot = get_umap(results, colors)

images_for_pdf.extend(visualizations)
images_for_pdf.append(umap_plot)
images_for_pdf.extend(mz_plots)
images_for_pdf.extend(diff_tables)

output_path = sys.argv[2]

images_for_pdf[0].save(os.path.join(output_path, "report.pdf"),
                       save_all=True,
                       append_images=images_for_pdf[1:])

for result in results:
    name = result["name"]
    w, h = result["components"]

    np.save(os.path.join(output_path, name + "_w.npy"), w)
    np.save(os.path.join(output_path, name + "_h.npy"), h)