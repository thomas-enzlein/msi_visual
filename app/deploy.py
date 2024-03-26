import streamlit as st
import glob
import os
import datetime
import json
import sys
from PIL import Image
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
import wx
import msi_visual
importlib.reload(msi_visual)
from msi_visual import nmf_segmentation
from msi_visual import visualizations
from msi_visual import objects
from msi_visual.app_utils.extraction_info import display_paths_to_extraction_paths, \
    get_files_from_folder
from msi_visual.utils import get_certainty
from msi_visual import parametric_umap
from msi_visual.umap_nmf_segmentation import SegmentationUMAPVisualization
importlib.reload(visualizations)

import hashlib
import base64

def make_hash_sha256(o):
    hasher = hashlib.md5()
    hasher.update(repr(make_hashable(o)).encode())
    return base64.urlsafe_b64encode(hasher.digest()).decode()

def make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k,make_hashable(v)) for k,v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))

    return o

def get_settings():
    return {'Extraction Root Folder': extraction_root_folder,
     'Extraction folder': selected_extraction,
     'Regions to include': regions,
     'Model folder': nmf_model_folder,
     'UMAP Model folder (optional)': umap_model_folder,
     'Segmentation model': nmf_model_name,
     'Image to show': image_to_show,
     'output_normalization': output_normalization,
     'color_schemes': st.session_state.color_schemes,
     'current_color_scheme': current_color_scheme}

def get_settings_hash():
    settings = get_settings()
    return make_hash_sha256(settings)

def save_data():
    if image_to_show:
        folder = datetime.datetime.today().strftime('%Y-%m-%d')
        os.makedirs(folder, exist_ok=True)
        image_name = Path(image_to_show).stem
        if umap_model_folder:
            prefix="umap"
        else:
            prefix="nmf"
        now = datetime.datetime.now()
        prefix = get_settings_hash() + "_" + prefix

        if 'latest_diff' in st.session_state:
            img, visualization, label_a, label_b = st.session_state.latest_diff
            Image.fromarray(img).save(Path(folder) / f"{image_name}_{label_a}_{label_b}_{prefix}.png")

            if 'latest_heatmaps' in st.session_state:
                for index, heatmap in enumerate(st.session_state.latest_heatmaps):
                    Image.fromarray(heatmap).save(Path(folder) / f"{image_name}_{[label_a, label_b][index]}_{prefix}_mask.png")
        else:
            visualization = st.session_state.results[image_to_show]["visualization"]
        Image.fromarray(visualization).save(Path(folder) / f"{image_name}_{prefix}.png")    

def save_to_cache(cache_path='deploy.cache'):    
    cache = get_settings()
    
    joblib.dump(cache, cache_path)

if os.path.exists("deploy.cache"):
    state = joblib.load("deploy.cache")
    cached_state = defaultdict(str)
    for k, v in state.items():
        cached_state[k] = v
else:
    cached_state = defaultdict(str)

app = wx.App()
wx.DisableAsserts()

if 'color_schemes' not in st.session_state:
    st.session_state.color_schemes = cached_state['color_schemes']
if st.session_state.color_schemes == '':
    st.session_state.color_schemes = ["gist_yarg"] * 100

results = {}
image_to_show = None
if 'run_id' not in st.session_state:
    st.session_state.run_id = 0
if 'bins' not in st.session_state:
    st.session_state.bins = 5    

if 'coordinates' not in st.session_state or st.session_state.coordinates is None:
    st.session_state.coordinates = defaultdict(list)

umap_model_folder, nmf_model_path, nmf_model_name = None, None, None
region_colorscheme = None
with st.sidebar:
    extraction_root_folder = st.text_input('Extraction Root Folder', value=cached_state["Extraction Root Folder"])
    if extraction_root_folder:
        extraction_folders = display_paths_to_extraction_paths(extraction_root_folder)
        extraction_folders_keys = list(extraction_folders.keys())
        if cached_state['Extraction folder'] == '':
            cached_state['Extraction folder'] = extraction_folders_keys[0]
        selected_extraction = st.selectbox('Extraction folder', extraction_folders_keys, index=extraction_folders_keys.index(cached_state['Extraction folder']))
        if selected_extraction:
            extraction_folder = extraction_folders[selected_extraction]
            region_list = get_files_from_folder(extraction_folder)
            if cached_state['Regions to include'] == '':
                default=None
            else:
                default = cached_state['Regions to include']
                if False in [r in region_list for r in cached_state['Regions to include'] ]:
                    default = None

            regions = st.multiselect('Regions to include', region_list,  default=default)
    
    st.divider()
    
    nmf_model_folder = st.text_input('Model folder', value=cached_state['Model folder'])
    if nmf_model_folder:
        nmf_model_paths = list(glob.glob(nmf_model_folder + "\\*.joblib")) \
            + list(glob.glob(nmf_model_folder + "\\*\\*.joblib"))
        nmf_model_display_paths = [Path(p).stem for p in nmf_model_paths]

        default = cached_state['Segmentation model']
        if default == '':
            default = None
        else:
            if default in nmf_model_display_paths:
                default = nmf_model_display_paths.index(default)
            else:
                default = None
        nmf_model_name = st.selectbox('Segmentation model', nmf_model_display_paths, index=default)

    umap_model_folder = st.text_input('UMAP Model folder (optional)', value=cached_state['UMAP Model folder (optional)'])
    output_normalization = st.selectbox('Segmentation Output Normalization', ['spatial_norm', 'None'])
    sub_sample = st.number_input('Subsample pixels', value=None)

    if 'results' in st.session_state:
        image_to_show = st.selectbox('Image to show', list(st.session_state.results.keys()), key = st.session_state.run_id)
    
    #objects_mode = st.checkbox('Objects mode')
    #objects_window = st.selectbox('Objects window size', [3, 6, 9, 12])
    objects_mode = False
    

if nmf_model_name:
    nmf_model_path = [p for p in nmf_model_paths if Path(p).stem == nmf_model_name][0]
    nmf = joblib.load(open(nmf_model_path, 'rb'))

if umap_model_folder:
    umap = parametric_umap.UMAPVirtualStain()
    umap.load(umap_model_folder)        
    seg_umap = SegmentationUMAPVisualization(umap, nmf)

with st.sidebar:
    st.divider()
    colorschemes = list(colormaps)
    if umap_model_folder:
        region_selectbox = st.selectbox(f"Region to control", [i for i in range(nmf.k)], index=0)
        
        default = colorschemes.index(st.session_state.color_schemes[0])
        if st.session_state.color_schemes != '' and len(st.session_state.color_schemes) > int(region_selectbox):
            default = colorschemes.index(st.session_state.color_schemes[int(region_selectbox)])
        region_colorscheme = st.selectbox(f"Color Scheme", colorschemes, index=default)

        if st.button('Update color scheme'):
            st.session_state.color_schemes[int(region_selectbox)] = region_colorscheme

        if st.button("Export color scheme"):
            dialog = wx.FileDialog(None, "Color scheme file location", style=wx.DD_DEFAULT_STYLE)
            if dialog.ShowModal() == wx.ID_OK:
                export_path = dialog.GetPath() # folder_path will contain the path of the folder you have selected as
                with open(export_path, 'w') as f:
                    json.dump(st.session_state.color_schemes, f)

        if st.button("Import color scheme"):
            dialog = wx.FileDialog(None, "Color scheme file location", style=wx.DD_DEFAULT_STYLE)
            if dialog.ShowModal() == wx.ID_OK:
                export_path = dialog.GetPath() # folder_path will contain the path of the folder you have selected as
                with open(export_path, 'r') as f:
                    st.session_state.color_schemes = json.load(f)
        
        current_color_scheme = str([str(x) for x in st.session_state.color_schemes])

    else:
        color_scheme = st.selectbox("Color Scheme", colorschemes, index = colorschemes.index("gist_rainbow"))
        current_color_scheme = color_scheme

start = st.button("Run")
#save = st.button("Save")
# if save:
#     save_data()

if start:
    save_to_cache()
    st.session_state.run_id = st.session_state.run_id + 1

    with st.spinner(text="Running segmentation.."):
        st.session_state.coordinates = None
        st.session_state.results = None
        st.session_state.difference_visualizations = None
        extraction_args = eval(open(Path(extraction_folder) / "args.txt").read())
        st.session_state.bins = extraction_args.bins
        st.session_state.extraction_start_mz = extraction_args.start_mz
        st.session_state.extraction_end_mz = extraction_args.end_mz

        for path in regions:
            if sub_sample:
                img = np.load(path)[::int(sub_sample), ::int(sub_sample), :]
            else:
                img = np.load(path)
            img = np.float32(img)

            if umap_model_folder:
                contributions = seg_umap.factorize(img, method=output_normalization)
            else:
                contributions = nmf.factorize(img)

            results[path] = {"mz_image": img, "data_for_visualization": contributions}
        
        st.session_state.results = results

    with st.sidebar:
        image_to_show = st.selectbox('Image to show', list(st.session_state.results.keys()), key = st.session_state.run_id)


if 'results' in st.session_state:
    for path, data in st.session_state.results.items():
        img, data_for_visualization = data["mz_image"], data["data_for_visualization"]
        if path in image_to_show:
            if umap_model_folder:
                nmf_segmentation_mask, _, _ = data_for_visualization
                _, segmentation_mask, visualization = seg_umap.visualize_factorization(img, data_for_visualization,
                                                                                    st.session_state.color_schemes, method=output_normalization)
            else:
                nmf_segmentation_mask = data_for_visualization
                segmentation_mask, visualization = nmf.visualize_factorization(img, data_for_visualization, color_scheme, method=output_normalization)
            st.session_state.results[path]["nmf_segmentation_mask"] = nmf_segmentation_mask

            certainty_image = None

            certainty_image = get_certainty(nmf_segmentation_mask, img)
            certainty_image[img.max(axis=-1) == 0] = 0
            
            if len(segmentation_mask.shape) > 2:
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

            if certainty_image is not None:
                st.session_state.results[path]["certainty"] = certainty_image
            point = streamlit_image_coordinates(visualization)

            if umap_model_folder:
                region_visualization = st.session_state.results[path]["visualization"].copy()
                mask = st.session_state.results[path]["nmf_segmentation_mask"] == int(region_selectbox)
                region_visualization[mask == 0] = 0
                st.image(region_visualization)

            if point is not None:
                if path in st.session_state.coordinates and point in [p for p in st.session_state.coordinates[path]]:
                    pass
                else:
                    st.session_state.coordinates[path].append(point)
                    st.session_state.coordinates[path] = st.session_state.coordinates[path][-2 : ]

            if certainty_image is not None:
                st.text('Clustering Certainty')
                st.image(np.uint8(255*certainty_image))

if image_to_show and st.session_state.coordinates and image_to_show in st.session_state.coordinates:
    images = []
    segmentation_mask = st.session_state.results[image_to_show]["segmentation_mask"]
    visualization = st.session_state.results[image_to_show]["visualization"]
    for point in st.session_state.coordinates[image_to_show]:
        x, y = int(point['x']), int(point['y'])
        heatmap = visualizations.get_mask(visualization, segmentation_mask, x, y)
        images.append(heatmap)
    st.session_state.latest_heatmaps = images
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

                    if 'current_color_scheme' not in st.session_state:
                        st.session_state['current_color_scheme'] = current_color_scheme

                    if image_to_show not in st.session_state.difference_visualizations or st.session_state['current_color_scheme'] != current_color_scheme:
                        save_to_cache()
                        diff = visualizations.RegionComparison(mz_image,
                                                               segmentation_mask,
                                                               visualization,
                                                               start_mz=st.session_state.extraction_start_mz,
                                                               bins_per_mz=st.session_state.bins)
                        st.session_state.difference_visualizations[image_to_show] = diff
                    else:
                        diff = st.session_state.difference_visualizations[image_to_show]

                    st.session_state['current_color_scheme'] = current_color_scheme
                    
                    if objects_mode:
                        point = st.session_state.coordinates[image_to_show][-1]
                        point = int(point['x']), int(point['y'])
                        image = diff.visualize_object_comparison(point, size=objects_window)
                        label_a, label_b = None, None
                        st.image(image)
                    else:
                        point_a = st.session_state.coordinates[image_to_show][0]
                        point_a = int(point_a['x']), int(point_a['y'])
                        point_b = st.session_state.coordinates[image_to_show][1]
                        point_b = int(point_b['x']), int(point_b['y'])
                        image, label_a, label_b = diff.visualize_comparison_between_points(point_a, point_b)
                        st.image(image)
                    st.session_state.latest_diff = (image, visualization, label_a, label_b)
    save_data()

if image_to_show:
    mz = st.text_input('Create ION image for m/z:')
    if mz:
        mz_image = st.session_state.results[image_to_show]["mz_image"]
        val = int( (float(mz)-st.session_state.extraction_start_mz) * st.session_state.bins)
        st.image(visualizations.create_ion_image(mz_image, val))