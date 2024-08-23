import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from PIL import Image
import cv2
from collections import defaultdict

import importlib
import utils.viewer
importlib.reload(utils.viewer)
from utils.viewer import display_comparison, get_stats

from msi_visual.utils import segment_visualization
from msi_visual.visualizations import get_mask
from msi_visual.extraction import get_extraction_mz_list
from msi_visual.normalization import total_ion_count

from dataclasses import dataclass
@dataclass
class ClickData:
    visualiation_path: str
    point: tuple[int, int]
    mask: np.ndarray
    visualization: np.ndarray
    

def get_data(path):
    if path in st.session_state["data"]:
        return st.session_state["data"][path]
    else:
        img = total_ion_count(np.load(path))
        st.session_state["data"][path] = img
        return img

def get_image(path, bins=5, equalize=False):
    key = "cache" + str(bins) + str(equalize)
    if key not in st.session_state:
        st.session_state[key] = {}

    if path in st.session_state[key]:
        img, segmentation = st.session_state[key][path]
    else:
        img = np.array(Image.open(path))
        if equalize:
            img = cv2.merge([cv2.equalizeHist(img[:, :, 0]), cv2.equalizeHist(img[:, :, 1]), cv2.equalizeHist(img[:, :, 2])])
        segmentation = segment_visualization(img, bins)
        st.session_state[key][path] = img, segmentation
    
    return img, segmentation

if "clicks" not in st.session_state:
    st.session_state["clicks"] = defaultdict(list)
    st.session_state["data"] = {}

viz_tab, region_tab, point_tab = st.tabs(["Visualizations", "Region/Region Comparisons", "Point/Point Comparisons"])

with viz_tab:
    folder = st.text_input("Visualization folder")
    bins = st.number_input("Number of segmentation bins", min_value=5, step=1, value=5)
    equalize = st.checkbox('Equalize')

    if folder:
        csv = pd.read_csv(Path(folder) / "visualization_details.csv")
        visualization_paths = csv.visualization
        N = 2
        cols = st.columns(2)
        rows = [visualization_paths[i : i + N] for i in range(0, len(visualization_paths), N)]
        for row in rows:        
            for col, path in zip(cols, row):
                visualization, seg = get_image(Path(folder) / path, bins, equalize)
                with col:
                    point = streamlit_image_coordinates(visualization)
                    data_path = csv[csv.visualization==path].data.values[0]
                    if point and point not in st.session_state["clicks"][data_path][-2 :]:
                        x, y = int(point['x']), int(point['y'])
                        mask = get_mask(visualization, seg, x, y)
                        st.session_state["clicks"][data_path].append(ClickData(path, (x, y), mask, visualization))


with region_tab:
    if 'ion' in st.session_state:
        st.image(st.session_state.ion)

    for data_path in st.session_state["clicks"]:
        st.session_state["clicks"][data_path] = st.session_state["clicks"][data_path][-2 : ]
        clicks = st.session_state["clicks"][data_path]
        extraction_mzs = get_extraction_mz_list(Path(data_path).parent)
        visualization_a = clicks[0].visualization

        if len(clicks) > 0:
            cols = st.columns(2)
            mask_a = clicks[0].mask
            cols[0].image(mask_a)
            if len(clicks) == 2:
                visualization_b = clicks[1].visualization             
                mask_b = clicks[1].mask
                cols[1].image(mask_b)
                data = get_data(data_path)

                mask_a = np.uint8(mask_a.max(axis=-1)) * 255
                mask_b = np.uint8(mask_b.max(axis=-1)) * 255

                stats = get_stats(data, extraction_mzs, clicks[0].visualiation_path, clicks[1].visualiation_path, mask_a, mask_b,
                    clicks[0].point, clicks[1].point)
                print(stats, clicks[0].point, clicks[1].point)
                display_comparison(stats, data, visualization_a, visualization_b, mask_a, mask_b, extraction_mzs)