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
from msi_visual.percentile_ratio import percentile_ratio_rgb
from msi_visual import nmf_3d
from msi_visual import parametric_umap
from msi_visual.app_utils.extraction_info import display_paths_to_extraction_paths, \
    get_files_from_folder
from keras.callbacks import Callback

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


for path in regions:
    if path in st.session_state.pr:
        pr = st.session_state.pr[path]
    else:
        with st.spinner(text=f"Generating High Saliency Percentile-Ratio Visualization for {path}.."):
            img = np.load(path)
            pr = percentile_ratio_rgb(img)
            st.session_state.pr[path] = pr
    
    st.text(path)
    st.image(pr)

    save_data(path=path)