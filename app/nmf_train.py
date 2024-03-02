import streamlit as st
import glob
import os
import json
import sys
import numpy as np
import joblib
from pathlib import Path
from msi_visual import nmf_segmentation
from PIL import Image
from st_pages import show_pages_from_config, add_page_title

# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
add_page_title()

show_pages_from_config()

paths = json.load(open(sys.argv[1]))

if 'bins' not in st.session_state:
    st.session_state.bins = 5


with st.sidebar:
    number_of_components = st.number_input('Number of components',  min_value=2, max_value=1000, value=10, step=1)
    start_mz = st.number_input('Start m/z', 0)
    end_mz = st.number_input('End m/z', value=None)
    output_path = st.text_input('Output path for segmentation model', 'models/nmf_model.joblib')
    sub_sample = st.number_input('Subsample pixels', value=None)

    folder = st.selectbox('Extration folder', paths.keys())
    if folder:
        regions = st.multiselect('Regions to include', paths[folder])

start = st.button("Train NMF segmentation")
if start:
    if start_mz is not None:
        start_bin = int(st.session_state.bins * (start_mz - 300))
    else:
        start_bin = 0
    
    if end_mz is not None:
        end_bin = int(st.session_state.bins * (end_mz - 300))
    else:
        end_bin = None

    seg = nmf_segmentation.NMFSegmentation(k=int(number_of_components), normalization='tic', start_bin=start_bin, end_bin=end_bin)

    if sub_sample:
        images = [np.load(p)[::int(sub_sample), ::int(sub_sample), :] for p in regions]
    else:
        images = [np.load(p) for p in regions]
    
    with st.spinner(text="Training NMF segmentation.."):
        st.session_state.bins = eval(open(Path(paths[folder][0]).parent / "args.txt").read()).bins
        seg.fit(images)

    joblib.dump(seg, output_path)

    for img in seg.visualize_training_components(images):
        st.image(img)
        