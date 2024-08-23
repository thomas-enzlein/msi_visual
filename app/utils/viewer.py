import numpy as np
import streamlit as st
from msi_visual import visualizations
import math
import cv2

def get_stats(data, extraction_mzs, visualization_path_a, visualization_path_b, mask_a, mask_b, point_a, point_b):
    print(point_a, point_b)
    key = visualization_path_a + visualization_path_b + str(point_a) + str(point_b)
    if 'stats' not in st.session_state:
        st.session_state['stats'] = {}
        
    if key in st.session_state['stats']:
        stats = st.session_state['stats'][key]
    else:
        comparison = visualizations.RegionComparison(
            data,
            mzs=extraction_mzs)
        stats = comparison.ranking_comparison(mask_a, mask_b)
        st.session_state['stats'][key] = stats
    return stats


def create_ion_image(img, mz, extraction_mzs):
    mz_index = np.abs(np.float32(extraction_mzs) - float(mz)).argmin()
    closest_mz = extraction_mzs[mz_index]
    if closest_mz != mz:
        st.text(f'Showing ION image for closest mz {closest_mz}')
    ion = visualizations.create_ion_image(img, mz_index)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (5, 20)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 1
    ion = cv2.putText(ion, f"{closest_mz:.3f}",
                         bottomLeftCornerOfText,
                         font,
                         fontScale,
                         fontColor,
                         thickness,
                         lineType)

    st.session_state.ion = ion

def display_comparison(stats, data, visualization_a, visualization_b, mask_a, mask_b, extraction_mzs):
    color_a = visualization_a[mask_a > 0].mean(axis=0)
    color_b = visualization_b[mask_b > 0].mean(axis=0)
    display_mzs(data, stats, color_a, color_b, extraction_mzs, threshold=0.5)

def display_mzs(img, stats, color_a, color_b, extraction_mzs, threshold=0.5, num_mzs=30):
    
    pos_aucs = {mz: -stats[mz]
                for mz in stats if stats[mz] < threshold}
    neg_aucs = {
        mz: stats[mz] for mz in stats if stats[mz] > (
            1 - threshold)}

    pos_aucs_keys = sorted(list(pos_aucs.keys()),
                           key=lambda x: pos_aucs[x])[-num_mzs:]
    neg_aucs_keys = sorted(list(neg_aucs.keys()),
                           key=lambda x: neg_aucs[x])[-num_mzs:]

    pos_aucs = {mz: pos_aucs[mz] for mz in pos_aucs_keys}
    neg_aucs = {mz: neg_aucs[mz] for mz in neg_aucs_keys}

    for auc_group, color in zip([pos_aucs, neg_aucs], [color_a, color_b]):
        header_cols = st.columns(2)
        header_cols[0].text(f'm/z values for ')
        header_cols[1].image(np.uint8(color * np.ones((64, 64, 3))))
        sorted_indices = sorted(
            list(
                auc_group.keys()),
            key=lambda mz: stats[mz],
            reverse=True)
        N = 8
        num_rows = math.ceil(len(sorted_indices) / N)
        for row in range(num_rows):
            indices = sorted_indices[N * row: N * row + N]
            indices = indices + [None] * (N - len(indices))
            for i, (mz, col) in enumerate(
                    zip(indices, st.columns(len(indices)))):
                if mz is None:
                    col.button(
                        "",
                        key=f"default+{i}+{row}+{col}+{mz}",
                        disabled=True,
                        use_container_width=True)
                else:
                    col.button(
                        f"{mz}",
                        use_container_width=True,
                        key=f"{mz}+{i}+{row}+{col}",
                        on_click=create_ion_image,
                        args=[img,
                            mz,
                            extraction_mzs])
