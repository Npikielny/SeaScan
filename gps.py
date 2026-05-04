from tqdm import tqdm
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
from loguru import logger

YAW_KEYS = [
    "GimbalYawDegree",
    "FlightYawDegree"
]
YAW_KEY = YAW_KEYS[0]

def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    
    if not exif_data:
        return None
    
    exif = {}
    for tag, value in exif_data.items():
        decoded = TAGS.get(tag, tag)
        exif[decoded] = value
    
    return exif

import subprocess
import json

def get_gimbal_pitch(image_path):
    cmd = [
        "exiftool",
        "-json",
        "-GimbalPitchDegree",
        "-CameraPitch",
        image_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)[0]
    
    return float(data.get("GimbalPitchDegree") or data.get("CameraPitch"))

def get_coords(data):
    if type(data) == str:
        exif = get_exif_data(data)
    else:
        exif = data
    gps = exif['GPSInfo']
    lat = gps[2]

    lon = gps[4]
    if gps[1] == 'S':
        lat = [-l for l in lat]
    if gps[3] == 'W':
        lon = [-l for l in lon]
    return np.array([
        lat,
        lon
    ]).astype(np.float64)


def get_gimbal_yaw(image_path):
    if YAW_KEY == "FlightYawDegree":
        cmd = [
            "exiftool",
            "-json",
            YAW_KEY,
            image_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)[0]

        yaw = float(data.get(YAW_KEY))
        return yaw
    else:
        cmd = [
            "exiftool",
            "-json",
            "-GimbalYawDegree",
            "-GimbalRollDegree",
            image_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)[0]

        yaw = float(data.get("GimbalYawDegree"))
        roll = float(data.get("GimbalRollDegree"))
        if roll == 180:
            yaw = 180 + yaw
        elif roll != 0:
            logger.warning(f"Found Camera Gimbal Roll != 0 or 180: {roll}")
        return yaw


def to_arc_seconds(data):
    return np.sum(data * [60 * 60, 60, 1], axis=-1)

def dist_from_centroid(data, target):
    return np.linalg.norm(to_arc_seconds(data) - to_arc_seconds(target), axis=-1)