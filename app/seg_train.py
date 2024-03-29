import streamlit as st
import glob
import os
import json
import sys
import numpy as np
import joblib
from pathlib import Path
from argparse import Namespace
from PIL import Image
from st_pages import show_pages_from_config, add_page_title

from msi_visual import nmf_segmentation, kmeans_segmentation
from msi_visual.app_utils.extraction_info import display_paths_to_extraction_paths, \
    get_files_from_folder

# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar

add_page_title()

show_pages_from_config()
if 'bins' not in st.session_state:
    st.session_state.bins = 5


with st.sidebar:    
    model_type = st.radio("Segmentation method", key="model",options=["NMF", "Kmeans"],)
    #model_type = st.selectbox("Model Type", ["NMF", "Kmeans"], index=0)
    
    extraction_root_folder = st.text_input("Extraction Root Folder", value="E:\\MSImaging-data\\")
    if extraction_root_folder:
        extraction_folders = display_paths_to_extraction_paths(extraction_root_folder)

        selected_extraction = st.selectbox('Extraction folder', extraction_folders.keys())
        if selected_extraction:
            extraction_folder = extraction_folders[selected_extraction]
            regions = st.multiselect('Regions to include', get_files_from_folder(extraction_folder))

            extraction_args = eval(open(Path(extraction_folder) / "args.txt").read())
            st.session_state.bins = extraction_args.bins
            st.session_state.extraction_start_mz = extraction_args.start_mz
            st.session_state.extraction_end_mz = extraction_args.end_mz

        start_mz = st.number_input('Start m/z', st.session_state.extraction_start_mz, step=50)
        #start_mz = st.number_input('Start m/z', min_value=0, value=300, step=50)
        end_mz = st.number_input('End m/z', min_value=100, value=None, step=50)
        number_of_components = st.number_input('Number of components (k)', min_value=2, max_value=100, value=5, step=5)
        output_path_default = "E:\\MSImaging-data\\_msi_visual\\" + model_type + "-models\\model.joblib"
        output_path = st.text_input('Output path for segmentation model', value=output_path_default)
        sub_sample = st.number_input('Subsample pixels', value=None)

start = st.button("Train " + model_type + " segmentation")
if start:

    extraction_args = eval(open(Path(extraction_folder) / "args.txt").read())
    st.session_state.bins = extraction_args.bins
    st.session_state.extraction_start_mz = extraction_args.start_mz
    st.session_state.extraction_end_mz = extraction_args.end_mz


    if start_mz is not None:
        start_bin = int(st.session_state.bins * (start_mz - st.session_state.extraction_start_mz))
    else:
        start_bin = st.session_state.extraction_start_mz
    
    if end_mz is not None:
        end_bin = int(st.session_state.bins * (end_mz - st.session_state.extraction_start_mz))
    else:
        end_bin = None

    if model_type == "NMF":
        seg = nmf_segmentation.NMFSegmentation(k=int(number_of_components), normalization='tic', start_bin=start_bin, end_bin=end_bin)
    else:
        seg = kmeans_segmentation.KmeansSegmentation(k=int(number_of_components), normalization='tic', start_bin=start_bin, end_bin=end_bin)

    if sub_sample:
        images = [np.load(p)[::int(sub_sample), ::int(sub_sample), :] for p in regions]
    else:
        images = [np.load(p) for p in regions]
    
    with st.spinner(text="Training segmentation.."):
        seg.fit(images)

    joblib.dump(seg, output_path)

    for img in seg.visualize_training_components(images):
        st.image(img)
        