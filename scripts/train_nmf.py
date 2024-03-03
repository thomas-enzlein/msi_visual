
from msi_visual import nmf_segmentation
import glob
import argparse
import numpy as np
from argparse import Namespace
from pathlib import Path
import joblib
import tqdm
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='',
                        help='Prefix to add to all model files')
    parser.add_argument('--input_path', type=str, required=True,
                        help='.d folder')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Where to store the output .npy files')
    parser.add_argument(
        '--number_of_components',
        type=list,
        default=[5, 10, 20, 40, 60, 80, 100],
        nargs='+',
        help='Number of components')
    parser.add_argument(
        '--start_mz', type=int, default=300,
        help='m/z to start from')
    parser.add_argument(
        '--end_mz', type=int, default=None,
        help='m/z to stop at')
    args = parser.parse_args()
    return args

args = get_args()

bins = eval(open(Path(args.input_path) / "args.txt").read()).bins
start_bin = int((args.start_mz - 300)*bins)
if args.end_mz is not None:
    end_bin = int((args.end_mz-300)*bins)
else:
    end_bin = None

paths = glob.glob(args.input_path + "/*.npy")
images = [np.load(p) for p in paths]

os.makedirs(args.output_path, exist_ok=True)

for k in tqdm.tqdm(args.number_of_components):
    seg = nmf_segmentation.NMFSegmentation(k=k, normalization='tic', start_bin=start_bin, end_bin=end_bin)
    seg.fit(images)
    
    if len(args.prefix) > 0:
        args.prefix = args.prefix + "_"
    name = f"{args.prefix}bins_{bins}_k_{k}_startmz_{args.start_mz}_end_mz_{args.end_mz}.joblib"
    output = Path(args.output_path) / name

    joblib.dump(seg, output)