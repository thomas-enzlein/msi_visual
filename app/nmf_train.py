import streamlit as st
import glob
import os
import json
import sys
import numpy as np
import joblib

from msi_visual import nmf_segmentation
from PIL import Image
from st_pages import show_pages_from_config, add_page_title

# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
add_page_title()

show_pages_from_config()

paths = json.load(open(sys.argv[1]))

with st.sidebar:
    number_of_components = st.selectbox('Number of components', list(range(2, 20)))
    start_bin = st.number_input('Start m/z bin', 0)
    end_bin = st.number_input('End m/z bin', value=None)
    output_path = st.text_input('Output path for segmentation model', 'models/nmf_model.joblib')
    sub_sample = st.number_input('Subsample pixels', value=None)

    folder = st.selectbox('Extration folder', paths.keys())
    if folder:
        regions = st.multiselect('Regions to include', paths[folder])

start = st.button("Train NMF segmentation")
if start:
    seg = nmf_segmentation.NMFSegmentation(k=int(number_of_components), normalization='tic', start_bin=start_bin, end_bin=end_bin)

    if sub_sample:
        images = [np.load(p)[::int(sub_sample), ::int(sub_sample), :] for p in regions]
    else:
        images = [np.load(p) for p in regions]
    
    with st.spinner(text="Training NMF segmentation.."):
        seg.fit(images)

    joblib.dump(seg, output_path)

    for img in seg.visualize_training_components(images):
        st.image(img)
        