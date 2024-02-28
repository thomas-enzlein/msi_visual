import streamlit as st
import glob
import os
import json
import sys
import numpy as np
import joblib
from msi_visual import nmf_segmentation
from matplotlib import colormaps
from matplotlib import pyplot as plt
from st_pages import show_pages_from_config, add_page_title
from streamlit_image_coordinates import streamlit_image_coordinates

paths = json.load(open(sys.argv[1]))
st.title('Visualize MSI with NMF')
results = []

with st.sidebar:
    model_path = st.selectbox('Segmentation model path', list(glob.glob("../models/*.joblib")))
    sub_sample = st.number_input('Subsample pixels', value=None)
    colorschemes = list(colormaps)
    color_scheme = st.selectbox("Color Scheme", colorschemes, index = colorschemes.index("gist_rainbow"))

    folder = st.selectbox('Extration folder', paths.keys())
    if folder:
        regions = st.multiselect('Regions to include', paths[folder])

if model_path:
    seg = joblib.load(open(model_path, 'rb'))

start = st.button("Run")
if start:
    with st.spinner(text="Running NMF segmentation.."):
        for path in regions:
            if sub_sample:
                img = np.load(path)[::int(sub_sample), ::int(sub_sample), :]
            else:
                img = np.load(path)

            contributions = seg.factorize(img)
            results.append((img, contributions))
        st.session_state.results = results
        st.session_state.color_scheme = color_scheme

coordinates = {}
if 'results' in st.session_state:
    for index, (img, contributions) in enumerate(st.session_state.results):
        _, vizualization = seg.visualize_factorization(img, contributions, color_scheme)
        coordinates[index] = streamlit_image_coordinates(vizualization)

for index in coordinates:
    if coordinates[index] is not None:
        x, y = coordinates[index]['x'], coordinates[index]['y']
        img, _ = st.session_state.results[index]
        mzs = np.float32(list(range(img.shape[-1]))) / 5
        fig = plt.figure()
        plt.plot(mzs, img[y, x, :])
        plt.xlabel('m/z')
        plt.title(f'Region: {index} xy = {x} {y}')
        plt.tight_layout()
        st.pyplot(fig)

