import sys, tsfdata, sqlite3
import numpy as np
from PIL import Image
import cv2
import tqdm
from bisect import bisect_left, bisect_right
import os
from multiprocessing import Pool

def _bisect_spectrum(mzs, mz_value, tol):
    ix_l, ix_u = bisect_left(mzs, mz_value - tol), bisect_right(mzs, mz_value + tol) - 1
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

def get_image(path, output_path, region=0, min_mz=300, max_mz=1300):
    tsf = tsfdata.TsfData(path)
    conn = tsf.conn
    xs = conn.execute(f"SELECT XIndexPos FROM MaldiFrameInfo WHERE RegionNumber={region}").fetchall()
    ys = conn.execute(f"SELECT YIndexPos FROM MaldiFrameInfo WHERE RegionNumber={region}").fetchall()
    regions = conn.execute("SELECT RegionNumber FROM MaldiFrameInfo").fetchall()
    xs, ys = np.int32(xs), np.int32(ys)
    xs = xs - np.min(xs)
    ys = ys - np.min(ys)
    width = np.max(xs) + 1
    height = np.max(ys) + 1
    regions = [r[0] for r in regions]
    start_index = regions.index(region)
    print(f"Start index for region {region} is {start_index}")
    num_mzs = max_mz-min_mz + 1
    img = np.zeros((height, width, num_mzs))
    
    for index, (x, y) in tqdm.tqdm(enumerate(zip(xs, ys)), total=len(xs)):
        indices, intensities = tsf.readLineSpectrum(start_index+index + 1)
        mzs = tsf.indexToMz(start_index+index+1, indices)
        for mz in range(num_mzs):
            min_i, max_i = _bisect_spectrum(mzs, mz, tol=0.1)
            img[y, x, mz] = np.sum(intensities[min_i:max_i+1])
    
    np.save(output_path, img)

def multi_run_wrapper(args):
   return get_image(*args)

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    tsf = tsfdata.TsfData(input_path)
    conn = tsf.conn
    regions = conn.execute(f"SELECT RegionNumber FROM MaldiFrameInfo").fetchall()
    regions = [r[0] for r in regions]
    regions = list(set(regions))

    args = [(input_path, os.path.join(output_path, f"{region}.npy"), region) \
            for region in regions]
    print(args)
    with Pool(4) as pool:
        pool.map(multi_run_wrapper, args)