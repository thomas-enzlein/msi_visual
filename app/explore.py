import streamlit as st
import glob
import os
import json
import sys
import numpy as np
import joblib
from collections import defaultdict
from pathlib import Path
from argparse import Namespace
from PIL import Image
from st_pages import show_pages_from_config, add_page_title
import importlib
import msi_visual.percentile_ratio
importlib.reload(msi_visual.percentile_ratio)
from msi_visual.percentile_ratio import percentile_ratio_rgb
from msi_visual import nmf_3d
from msi_visual import parametric_umap
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

# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
add_page_title()

show_pages_from_config()

if 'pr' not in st.session_state:
    st.session_state.pr = {}

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
        if path+settings_str in st.session_state.pr:
            pr = st.session_state.pr[path+settings_str]
        else:
            with st.spinner(text=f"Generating High Saliency Percentile-Ratio Visualization for {path}.."):
                img = np.load(path)
                pr = percentile_ratio_rgb(img, percentiles=percentiles, equalize=equalize)
                st.session_state.pr[path + settings_str] = pr
        
        st.text(path)
        st.image(pr)

        save_data(path=path+settings_str)