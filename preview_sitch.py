import cv2, numpy as np, glob
from tqdm import tqdm
from PIL import Image
import warping

import gps, registration

from loguru import logger
import tyro

log = logger.debug

from pathlib import Path
import os

from project import Project

import rasterio
from rasterio.transform import Affine

from calib import Calibration

import html_map

def main(
    root: str,
    skip: int = 1   
):
    p = Path(root)
    imgs = glob.glob(str(p / "*.tif"))
    html_map.create_map(imgs, p, skip)

if __name__ == "__main__":
    tyro.cli(main)