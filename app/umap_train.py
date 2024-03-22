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

from msi_visual import nmf_segmentation
from msi_visual import parametric_umap
from msi_visual.app_utils.extraction_info import display_paths_to_extraction_paths, \
    get_files_from_folder
from keras.callbacks import Callback

trainprogress_bar = st.progress(0, text="Training UMAP..")
trainprogress_bar.empty()

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        global train_progress
        train_progress = train_progress + 0.01
        trainprogress_bar.progress(int(train_progress * 100), "Training UMAP..")


# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
add_page_title()

show_pages_from_config()


if 'bins' not in st.session_state:
    st.session_state.bins = 5

with st.sidebar:
    extraction_root_folder = st.text_input("Extraction Root Folder")
    if extraction_root_folder:
        extraction_folders = display_paths_to_extraction_paths(extraction_root_folder)

        selected_extraction = st.selectbox('Extration folder', extraction_folders.keys())
        if selected_extraction:
            extraction_folder = extraction_folders[selected_extraction]
            regions = st.multiselect('Regions to include', get_files_from_folder(extraction_folder))

            extraction_args = eval(open(Path(extraction_folder) / "args.txt").read())
            st.session_state.bins = extraction_args.bins
            st.session_state.extraction_start_mz = extraction_args.start_mz
            st.session_state.extraction_end_mz = extraction_args.end_mz

    start_mz = st.number_input('Start m/z', st.session_state.extraction_start_mz)
    end_mz = st.number_input('End m/z', value=None)
    output_path = st.text_input('Output Folder')
    sub_sample = st.number_input('Subsample pixels', value=None)


start = st.button("Train 1D Parametric UMAP")
if start:
    if start_mz is not None:
        start_bin = int(st.session_state.bins * (start_mz - st.session_state.extraction_start_mz))
    else:
        start_bin = st.session_state.extraction_start_mz
    
    if end_mz is not None:
        end_bin = int(st.session_state.bins * (end_mz - st.session_state.extraction_start_mz))
    else:
        end_bin = None

    umap = parametric_umap.UMAPVirtualStain(start_bin=start_bin, end_bin=end_bin)

    if sub_sample:
        images = [np.load(p)[::int(sub_sample), ::int(sub_sample), :] for p in regions]
    else:
        images = [np.load(p) for p in regions]
    
    with st.spinner(text="Training UMAP.."):
        train_progress = 0.0

        trainprogress_bar = st.progress(0, text="Training UMAP..")


        keras_fit_kwargs = {"callbacks": [MyCallback()]}
        umap.fit(images, keras_fit_kwargs=keras_fit_kwargs)
        trainprogress_bar.empty()

        umap.save(output_path)