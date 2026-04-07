from PIL import Image
from loguru import logger
log = loguru.debug
import numpy as np

import gps

# def overlaps(centroid, size):

def stitch_map(coordinates, images, transform):
    # Find bounds
    
    