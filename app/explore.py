import streamlit as st
import glob
import os
import json
import sys
import numpy as np
import time
import joblib
from collections import defaultdict
from pathlib import Path
from argparse import Namespace
from PIL import Image
from st_pages import show_pages_from_config, add_page_title
import importlib
from msi_visual.normalization import total_ion_count
import msi_visual.percentile_ratio
importlib.reload(msi_visual.percentile_ratio)
from msi_visual.percentile_ratio import percentile_ratio_rgb, top3
from msi_visual.outliers import get_outlier_image
from msi_visual.metrics import MSIVisualizationMetrics
from msi_visual.app_utils.extraction_info import display_paths_to_extraction_paths, \
    get_files_from_folder

def save_data(path=None):
    folder = "percentile_ratio_images"
    os.makedirs(folder, exist_ok=True)
    if path is None:
        paths = list(st.session_state.pr.keys())
    else:
        paths = [path]
    for path in paths:
        image_name = path.replace("/", "_").replace("\\", "_").replace(".", "_").replace(":", "_")
        Image.fromarray(st.session_state.pr[path]).save(Path(folder) /
                                f"{image_name}_pr.png")
        
        image_name = path.replace("/", "_").replace("\\", "_").replace(".", "_").replace(":", "_")
        Image.fromarray(st.session_state.max_intensity[path]).save(Path(folder) /
                                f"{image_name}_top3.png")

# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
add_page_title()

show_pages_from_config()

if 'pr' not in st.session_state:
    st.session_state.pr = {}

if 'outliers' not in st.session_state:
    st.session_state.outliers = {}

if 'max_intensity' not in st.session_state:
    st.session_state.max_intensity = {}

if 'pr_metrics' not in st.session_state:
    st.session_state.pr_metrics = {}

if 'max_intensity_metrics' not in st.session_state:
    st.session_state.max_intensity_metrics = {}

regions = []
with st.sidebar:
    extraction_root_folder = st.text_input("Extraction Root Folder")

    if extraction_root_folder:
        extraction_folders = display_paths_to_extraction_paths(extraction_root_folder)

        selected_extraction = st.selectbox('Extraction folder', extraction_folders.keys())
        if selected_extraction:
            extraction_folder = extraction_folders[selected_extraction]
            paths=get_files_from_folder(extraction_folder)
            regions = st.multiselect('Regions to include', paths, paths)


cols = st.columns(6)
st.text("Will be computed as Percentile(a)/Percentile(b), Percentile(c)/Percentile(d), Percentile(e)/Percentile(f)")
p0 = cols[0].number_input("a", min_value=0.0, max_value=100.0, value=99.99)
p1 = cols[1].number_input("b", min_value=0.0, max_value=100.0, value=99.9)
p2 = cols[2].number_input("c", min_value=0.0, max_value=100.0, value=99.9)
p3 = cols[3].number_input("d", min_value=0.0, max_value=100.0, value=99.0)
p4 = cols[4].number_input("e", min_value=0.0, max_value=100.0, value=98.0)
p5 = cols[5].number_input("f", min_value=0.0, max_value=100.0, value=85.0)

percentiles = [p0, p1, p2, p3, p4, p5]

equalize = st.checkbox("Equalize histogram internally")

settings_str = "".join([f"{p:.3f}" for p in percentiles]) + str(equalize)

if st.button("Run"):
    for path in regions:
        key = path+settings_str
        if key in st.session_state.pr:
            pr = st.session_state.pr[key]
            max_intensity = st.session_state.max_intensity[key]
            outlier = st.session_state.outliers[key]
            pr_metrics = st.session_state.pr_metrics[key]
            max_intensity_metrics = st.session_state.max_intensity_metrics[key]

            st.image(outlier)
            st.image(pr)
            st.image(max_intensity)
            st.write(pr_metrics)
            st.write(max_intensity_metrics)

        else:
            with st.spinner(text=f"Generating High Saliency Visualizations for {path}.."):
                st.text(path)
                t0 = time.time()
                img = np.load(path)
                img = total_ion_count(img)
                t1 = time.time()
                print(f"Loading took {t1-t0}")
                pr = percentile_ratio_rgb(img, percentiles=percentiles, equalize=equalize, normalization=None)
                t2 = time.time()
                print(f"PR tool {t2-t1}")
                st.image(pr)
                max_intensity = top3(img, normalization=None)
                st.image(max_intensity)
                pr_metrics = MSIVisualizationMetrics(img, pr, num_samples=3000).get_metrics()
                max_intensity_metrics = MSIVisualizationMetrics(img, max_intensity, num_samples=3000).get_metrics()
                t3 = time.time()
                print(f"Metrics took {t3-t2}")
                st.write("Percentile ratio metrics")
                st.write(pr_metrics)
                st.write("TOP-3 metrics")
                st.write(max_intensity_metrics)

                outlier = get_outlier_image(img)
                st.image(outlier)
                t4 = time.time()
                print(f"Outliers took {t4-t3}")

                st.session_state.pr_metrics[key] = pr_metrics
                st.session_state.max_intensity_metrics[key] = max_intensity_metrics
                st.session_state.pr[key] = pr
                st.session_state.max_intensity[key] = max_intensity
        
        


        save_data(path=path+settings_str)