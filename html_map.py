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

def save_geotiff(target, v_mask, M, destination, step, downsample=4, calibration=None):
    tiff = (calibration.open(target) * v_mask[..., np.newaxis]).astype(np.uint8)
    for _ in range(downsample):
        tiff = cv2.pyrDown(tiff)
    height, width = tiff.shape[:2]
    
    C = gps.to_arc_seconds(gps.get_coords(target))
    frame = registration.get_image_transform(gps.get_gimbal_yaw(target), tiff.shape)
    
    mat = np.vstack([
        frame[1].T, [0, 0]
    ])
    mat = np.hstack([
        mat,
        np.hstack([frame[0], [1]]).reshape((3, -1))
    ])
    # GPSs

    downsample = 2 ** downsample
    mat2 = np.vstack([
        M.T / downsample, [0, 0]
    ])
    mat2 = np.hstack([
        mat2,
        np.array([0, 0, 1]).reshape(3, 1)
    ])
    mat3 = np.array([
        [1, 0, -C[0]],
        [0, 1, -C[1]],
        [0, 0, 1]
    ])
    mat3 = mat3 @ np.array([
        [3600, 0, 0],
        [0, 3600, 0],
        [0, 0, 1]
    ])
    MAT = mat @ mat2 @ mat3
    
    mat = np.linalg.inv(MAT)
    mat = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ]) @ mat

    transform = Affine(
        mat[0, 0], mat[0, 1], mat[0, 2],
        mat[1, 0], mat[1, 1], mat[1, 2]
    )

    def prep_raster_for_rasterio(img):
        # img: (1, H, W, C) → (C, H, W)
        # img = img.squeeze(0)          # remove batch dim
        img = img.transpose(2, 0, 1)  # channels last → first
        return img
    
    img = tiff
    Y, X = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    ctr = np.array(img.shape[:2])[::-1] / 2
    UV = (np.dstack([Y, X]) - ctr) / ctr
    alpha2 = np.exp(-np.vecdot(UV, UV) * 6 / step)
    
    # alpha2 = D / D.max()# < 0.33
    # alpha2 = 1 - alpha2
    # alpha2 *= alpha2
    # alpha2 *= alpha2

    # import matplotlib.pyplot as plt
    # plt.imshow(alpha2); plt.colorbar(); plt.show(); 
    
    MAX_ALPHA = 255
    # alpha = np.clip((alpha * alpha2 * MAX_ALPHA), 0, 255).astype(np.uint8)
    alpha = np.clip(alpha2 * MAX_ALPHA, 0, 255).astype(np.uint8)
    
    # ---- WRITE GEOTIFF ----
    DEST = str(destination / f"output_{Path(target).stem}.tif")
    with rasterio.open(
        DEST,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,  # number of bands
        dtype=tiff.dtype,
        crs="EPSG:4326",  # change if needed
        transform=transform,
    ) as dst:
        d = np.vstack([
                prep_raster_for_rasterio(tiff[::-1, ::-1]),
                alpha[np.newaxis, ...]
            ])
        dst.write(
            d
        )
    return DEST
    
    
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import reshape_as_image
import folium
from folium.raster_layers import ImageOverlay

import webbrowser
def create_map(geotiffs, destination, skip):
    images = []
    bounds = []
    for r in geotiffs:
        with rasterio.open(r) as src:
            img = reshape_as_image(src.read())  # converts (bands, H, W) -> (H, W, bands)
            images.append(img)
            bounds.append(src.bounds)

    images = []
    bounds = []

    for r in geotiffs:
        with rasterio.open(r) as src:
            # print("SKEWED RASTER")
            # Check if the raster is skewed
            if src.transform.b != 0 or src.transform.d != 0:
                # print("Raster is rotated/skewed. Reprojecting to north-up...")
                
                # Compute transform for axis-aligned version
                transform, width, height = calculate_default_transform(
                    src.crs, src.crs, src.width, src.height, *src.bounds
                )
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': src.crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })
                
                # Create a temporary in-memory raster
                # reprojected_raster = np.zeros((src.count, height, width), dtype=src.dtypes[0])
                # reprojected_raster.fill(src.nodata)
                reprojected_raster = np.full((src.count, height, width), 0, dtype=src.dtypes[0])
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=reprojected_raster[i-1],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        resampling=Resampling.bilinear,
                        src_nodata=src.nodata,
                        dst_nodata=src.nodata
                    )
                img = reshape_as_image(reprojected_raster)
                Y, X = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
                ctr = np.array(img.shape[:2])[::-1] / 2
                UV = (np.dstack([Y, X]) - ctr) / ctr
                alpha2 = np.exp(-np.vecdot(UV, UV) * 6 / skip)
                
                alpha = np.where(np.all(reprojected_raster == 0, axis=0), 0, 1)
                # alpha2 = D / D.max()# < 0.33
                # alpha2 = 1 - alpha2
                # alpha2 *= alpha2
                # alpha2 *= alpha2

                # import matplotlib.pyplot as plt
                # plt.imshow(alpha2); plt.colorbar(); plt.show(); 
                
                MAX_ALPHA = 255
                # alpha = np.clip((alpha * alpha2 * MAX_ALPHA), 0, 255).astype(np.uint8)
                alpha = np.clip(alpha * alpha2 * MAX_ALPHA, 0, 255).astype(np.uint8)
                # plt.imshow(alpha); plt.colorbar(); plt.show()
                    #.astype(np.uint8)
                if img.shape[2] == 3:
                    img = np.dstack([img, alpha])

                bounds.append([
                    transform.f + height * transform.e,  # south
                    transform.c,                          # west
                    transform.f,                           # north
                    transform.c + width * transform.a      # east
                ])
            else:
                # print("Raster is north-up.")
                img = reshape_as_image(src.read())
                bounds.append([src.bounds.bottom, src.bounds.left, src.bounds.top, src.bounds.right])

            images.append(img)

    # Create Folium map centered on first raster
    m = folium.Map(location=[(bounds[0][2] + bounds[0][0]) / 2,
                            (bounds[0][3] + bounds[0][1]) / 2], zoom_start=17, max_zoom=30)

    # Add basemap
    folium.TileLayer(
        tiles='https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg',
        attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors',
        name='Stamen Terrain'
    ).add_to(m)

    # Add rasters as overlays
    for im, b in tqdm(zip(images, bounds), "Adding Tiles", total=len(images)):
        ImageOverlay(
            image=im,
            bounds=[[b[0], b[1]], [b[2], b[3]]],
            opacity=0.7,
        ).add_to(m)

    log("Saving map...")
    dest = destination / "map.html"
    m.save(str(dest))
    log(f"Map saved as {str(destination)}/map.html")

    webbrowser.open('file://' + str(dest.resolve()))

def execute_thread(images, v_mask, M, map_path, step, downsample, calibration, thread_id):
    for image in tqdm(images, f"Thread {thread_id}"):
        save_geotiff(image, v_mask, M, map_path, step, downsample, calibration)


from threading import Thread
from shutil import rmtree
def main(
    working: str,
    downsample: int = 4,
    step: int = 1,
    calibration: str | None = "./calibration.npy",
    project_name: str | None = None,
    mask: bool = True,
    threads: int = 6
):
    if project_name is None:
        project = Path(working) / "project.npy"
    else:
        project = Path(project_name)

    if not project.exists():
        raise "Unable to load project file because it does not exist: " + str(project)
    proj = Project(str(project))
    images = sorted(proj['images'])[::step]
    target = proj['target']
    v_mask = proj['v_mask']
    if not mask:
        v_mask = np.ones(v_mask.shape)
    M = proj['transform']

    map_path = Path(working) / "map_sources"
    if not map_path.exists():
        os.mkdir(str(map_path))

    calib = Calibration(calibration)

    n_per_thread = int(np.ceil(len(images) / threads))
    threads = []
    for ID, i in enumerate(range(0, len(images), n_per_thread)):
        t = Thread(
            target=execute_thread,
            args=(images[i: min(i + n_per_thread, len(images))], v_mask, M, map_path, step, downsample, calib, ID + 1)
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    geotiffs = glob.glob(str(map_path / "*.tif"))

    create_map(geotiffs, Path(working), skip=step)
    rmtree(str(map_path))
    

    
    



if __name__ == "__main__":
    tyro.cli(main)