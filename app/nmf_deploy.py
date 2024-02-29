import streamlit as st
import glob
import os
import json
import sys
import numpy as np
import joblib
from matplotlib import colormaps
from matplotlib import pyplot as plt
from collections import defaultdict
from st_pages import show_pages_from_config, add_page_title
from streamlit_image_coordinates import streamlit_image_coordinates

import importlib
import msi_visual
importlib.reload(msi_visual)
from msi_visual import nmf_segmentation
from msi_visual import visualizations

importlib.reload(visualizations)

paths = json.load(open(sys.argv[1]))
st.title('Visualize MSI with NMF')
results = {}
image_to_show = []
with st.sidebar:
    model_path = st.selectbox('Segmentation model path', list(glob.glob("../models/*.joblib")))
    sub_sample = st.number_input('Subsample pixels', value=None)
    colorschemes = list(colormaps)
    color_scheme = st.selectbox("Color Scheme", colorschemes, index = colorschemes.index("gist_rainbow"))
    folder = st.selectbox('Extration folder', paths.keys())
    if folder:
        regions = st.multiselect('Regions to include', paths[folder])

    if 'results' in st.session_state:
        image_to_show = st.selectbox('Image to show', list(st.session_state.results.keys()))

if model_path:
    seg = joblib.load(open(model_path, 'rb'))

start = st.button("Run")
if start:
    with st.spinner(text="Running NMF segmentation.."):
        st.session_state.coordinates = None
        st.session_state.results = None
        st.session_state.difference_visualizations = None

        for path in regions:
            if sub_sample:
                img = np.load(path)[::int(sub_sample), ::int(sub_sample), :]
            else:
                img = np.load(path)

            contributions = seg.factorize(img)
            results[path] = {"mz_image": img, "contributions": contributions}
        
        st.session_state.results = results
        st.session_state.color_scheme = color_scheme
    with st.sidebar:
        image_to_show = st.selectbox('Image to show', list(st.session_state.results.keys()))

if 'coordinates' not in st.session_state or st.session_state.coordinates is None:
    st.session_state.coordinates = defaultdict(list)

if 'results' in st.session_state:
    for path, data in st.session_state.results.items():
        img, contributions = data["mz_image"], data["contributions"]
        if path in image_to_show:
            segmentation_mask, visualization = seg.visualize_factorization(img, contributions, color_scheme)
            segmentation_mask = segmentation_mask.argmax(axis=0)
            segmentation_mask[img.max(axis=-1) == 0] = -1
            st.session_state.results[path]["segmentation_mask"] = segmentation_mask
            st.session_state.results[path]["visualization"] = visualization
            point = streamlit_image_coordinates(visualization)
            if point is not None:
                if path in st.session_state.coordinates and point in [p[0] for p in st.session_state.coordinates[path]]:
                    pass
                else:
                    st.session_state.coordinates[path].append((point, segmentation_mask, visualization))
                    st.session_state.coordinates[path] = st.session_state.coordinates[path][-2 : ]

if image_to_show and st.session_state.coordinates and image_to_show in st.session_state.coordinates:
    images = []
    for point, segmentation_mask, visualization in st.session_state.coordinates[image_to_show]:
        x, y = int(point['x']), int(point['y'])
        heatmap = visualizations.get_mask(visualization, segmentation_mask, x, y)
        images.append(heatmap)
    with st.container():
        for index, col in enumerate(st.columns(len(images))):
            col.image(images[index])
        if len(st.session_state.coordinates[image_to_show]) > 1:
            with st.spinner(text="Comparing regions.."):
                mz_image = st.session_state.results[image_to_show]["mz_image"]
                segmentation_mask = st.session_state.results[image_to_show]["segmentation_mask"]
                visualization = st.session_state.results[image_to_show]["visualization"]

                if 'difference_visualizations' not in st.session_state or st.session_state.difference_visualizations is None:
                    st.session_state['difference_visualizations'] = {}

                if image_to_show not in st.session_state.difference_visualizations:
                    diff = visualizations.RegionComparison(mz_image, segmentation_mask, visualization, start_mz=300, bins_per_mz=5)
                    st.session_state.difference_visualizations[image_to_show] = diff
                else:
                    diff = st.session_state.difference_visualizations[image_to_show]
                
                point_a = st.session_state.coordinates[image_to_show][0][0]
                point_a = int(point_a['x']), int(point_a['y'])
                point_b = st.session_state.coordinates[image_to_show][1][0]
                point_b = int(point_b['x']), int(point_b['y'])
                image = diff.visualize_comparison_between_points(point_a, point_b)
                st.image(image)

if image_to_show:
    mz = st.text_input('Create ION image for m/z:')
    if mz:
        mz_image = st.session_state.results[image_to_show]["mz_image"]
        val = int( (float(mz)-300) * 5)
        st.image(visualizations.create_ion_image(mz_image, val))