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
from argparse import Namespace
from st_pages import show_pages_from_config, add_page_title
from streamlit_image_coordinates import streamlit_image_coordinates
from pathlib import Path
import importlib
import msi_visual
importlib.reload(msi_visual)
from msi_visual import nmf_segmentation
from msi_visual import visualizations
from msi_visual import objects

importlib.reload(visualizations)

paths = json.load(open(sys.argv[1]))
st.title('NMF-Stain visualization for MSI')
results = {}
image_to_show = []

if 'run_id' not in st.session_state:
    st.session_state.run_id = 0

if 'bins' not in st.session_state:
    st.session_state.bins = 5

with st.sidebar:
    model_path = st.selectbox('Segmentation model path', list(glob.glob("../models/*.joblib")))
    sub_sample = st.number_input('Subsample pixels', value=None)
    colorschemes = list(colormaps)
    color_scheme = st.selectbox("Color Scheme", colorschemes, index = colorschemes.index("gist_rainbow"))
    folder = st.selectbox('Extration folder', paths.keys())
    if folder:
        regions = st.multiselect('Regions to include', paths[folder])

    if 'results' in st.session_state:
        image_to_show = st.selectbox('Image to show', list(st.session_state.results.keys()), key = st.session_state.run_id)
    
    objects_mode = st.checkbox('Objects mode')
    objects_window = st.selectbox('Objects window size', [3, 6, 9, 12])
    

if model_path:
    seg = joblib.load(open(model_path, 'rb'))

start = st.button("Run")
if start:
    st.session_state.run_id = st.session_state.run_id + 1
    with st.spinner(text="Running NMF segmentation.."):
        st.session_state.coordinates = None
        st.session_state.results = None
        st.session_state.difference_visualizations = None

        st.session_state.bins = eval(open(Path(paths[folder][0]).parent / "args.txt").read()).bins

        for path in regions:
            if sub_sample:
                img = np.load(path)[::int(sub_sample), ::int(sub_sample), :]
            else:
                img = np.load(path)
            img = np.float32(img)

            contributions = seg.factorize(img)
            results[path] = {"mz_image": img, "contributions": contributions}
        
        st.session_state.results = results
        st.session_state.color_scheme = color_scheme

    with st.sidebar:
        image_to_show = st.selectbox('Image to show', list(st.session_state.results.keys()), key = st.session_state.run_id)

if 'coordinates' not in st.session_state or st.session_state.coordinates is None:
    st.session_state.coordinates = defaultdict(list)

if 'results' in st.session_state:
    for path, data in st.session_state.results.items():
        img, contributions = data["mz_image"], data["contributions"]
        if path in image_to_show:
            segmentation_mask, visualization = seg.visualize_factorization(img, contributions, color_scheme)
            segmentation_mask = segmentation_mask.argmax(axis=0)
            segmentation_mask[img.max(axis=-1) == 0] = -1

            if objects_mode:
                with st.spinner(text="Detecting objects.."):

                    key = str(objects_window) + "_" + path
                    if 'object_mask' in st.session_state and key in st.session_state.object_mask:
                        object_mask = st.session_state.object_mask[key]
                    else:
                        if 'object_mask' not in st.session_state:
                            st.session_state.object_mask = {}
                            
                        detector = objects.ObjectDetector(size=int(objects_window))
                        object_mask = detector.get_mask(img)
                        st.session_state.object_mask[key] = object_mask

                    segmentation_mask[object_mask == 0] = -1
                    visualization[object_mask == 0] = 0

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
        if (objects_mode and len(st.session_state.coordinates[image_to_show]) > 0) or \
            ((not objects_mode) and len(st.session_state.coordinates[image_to_show]) > 1):
                with st.spinner(text="Comparing regions.."):
                    mz_image = st.session_state.results[image_to_show]["mz_image"]
                    segmentation_mask = st.session_state.results[image_to_show]["segmentation_mask"]
                    visualization = st.session_state.results[image_to_show]["visualization"]

                    if 'difference_visualizations' not in st.session_state or st.session_state.difference_visualizations is None:
                        st.session_state['difference_visualizations'] = {}

                    if image_to_show not in st.session_state.difference_visualizations:
                        diff = visualizations.RegionComparison(mz_image, segmentation_mask, visualization, start_mz=300, bins_per_mz=st.session_state.bins)
                        st.session_state.difference_visualizations[image_to_show] = diff
                    else:
                        diff = st.session_state.difference_visualizations[image_to_show]
                    
                    if objects_mode:
                        point = st.session_state.coordinates[image_to_show][-1][0]
                        point = int(point['x']), int(point['y'])
                        image = diff.visualize_object_comparison(point, size=objects_window)
                        st.image(image)
                    else:
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
        val = int( (float(mz)-300) * st.session_state.bins)
        st.image(visualizations.create_ion_image(mz_image, val))