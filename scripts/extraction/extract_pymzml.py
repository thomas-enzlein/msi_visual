import sys
import numpy as np
from PIL import Image
import cv2
import tqdm
from bisect import bisect_left, bisect_right
from typing import Optional
import os
import math
import os
import argparse
from pyimzml.ImzMLParser import ImzMLParser


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_mz', type=int, default=300)
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--end_mz', type=int, default=1300)
    parser.add_argument('--input_path', type=str, required=True,
                        help='.d folder')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Where to store the output .npy files')
    parser.add_argument(
        '--tol',
        type=float,
        default=None,
        help='If not None, will be used for binary search with tolerance')
    parser.add_argument(
        '--bins', type=int, default=1,
        help='How many bins per m/z value')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Parallel processes')
    args = parser.parse_args()
    return args


def _bisect_spectrum(mzs, mz_value, tol):
    ix_l, ix_u = bisect_left(
        mzs, mz_value - tol), bisect_right(mzs, mz_value + tol) - 1
    if ix_l == len(mzs):
        return len(mzs), len(mzs)
    if ix_u < 1:
        return 0, 0
    if ix_u == len(mzs):
        ix_u -= 1
    if mzs[ix_l] < (mz_value - tol):
        ix_l += 1
    if mzs[ix_u] > (mz_value + tol):
        ix_u -= 1
    return ix_l, ix_u


def get_image(
        path: str,
        output_path: str,
        region: int = 0,
        min_mz: int = 300,
        max_mz: int = 1350,
        tol: Optional[float] = None,
        bins_per_mz: int = 1):
    
    p = ImzMLParser(path)
    my_spectra = []
    
    xs = []
    ys = []
    all_mzs = []
    all_intensities = []
    print(len(p.coordinates))
    for idx, (x,y,z) in tqdm.tqdm(enumerate(p.coordinates), total=len(p.coordinates)):
        mzs, intensities = p.getspectrum(idx)
        xs.append(x)
        ys.append(y)
        all_mzs.append(mzs)
        all_intensities.append(intensities)

    xs, ys = np.int32(xs), np.int32(ys)
    xs = xs - np.min(xs)
    ys = ys - np.min(ys)
    width = np.max(xs) + 1
    height = np.max(ys) + 1
    num_mzs = max_mz - min_mz + 1
    img = np.zeros((height, width, bins_per_mz * num_mzs), dtype=np.float32)

    for x, y, mzs, intensities in tqdm.tqdm(zip(xs, ys, all_mzs, all_intensities)):
        intensities = np.float32(intensities)
        bins = np.int32(np.round((np.float32(mzs) - min_mz) * bins_per_mz))
        img[y, x, bins] = img[y, x, bins] + intensities

    np.save(output_path, img)
    return min_mz, max_mz

if __name__ == "__main__":
    args = get_args()
    
    print(args)
    input_path = args.input_path
    output_path = args.output_path

    args_description = str(args)
    with open(os.path.join(output_path, "args.txt"), "w") as f:
        f.write(args_description)

    os.makedirs(output_path, exist_ok=True)
    

    get_image(input_path,
         os.path.join(
             output_path,
             f"{0}.npy"),
            0,
            args.start_mz,
            args.end_mz,
            args.tol,
            args.bins)
    
