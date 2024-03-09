from argparse import Namespace
from pathlib import Path
import glob
import os

def display_path(path):
    try:
        metadata = eval(open(Path(path) / "args.txt").read())
        bins = metadata.bins
        if 'id' in metadata:
            id = metadata.id + ' '
        else:
            id = ''
        return f"{id}m/z {metadata.start_mz}-{metadata.end_mz} {bins} bins"
    except Exception as e:
        return None

def display_paths_to_extraction_paths(extraction_paths):
    result =  dict(zip([display_path(p) for p in extraction_paths], extraction_paths))
    return {k: v for k, v in result.items() if v is not None}

def get_files_from_folder(path):
    return glob.glob(os.path.join(path, "*.npy"))