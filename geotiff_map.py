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

def get_image_corners(image, frame, center, M, shape):
    corners = np.array([
        0, 0,
        0, shape[1],
        shape[0], 0,
        shape[0], shape[1],
    ]).reshape((-1, 2))

    res = np.array([registration.image_to_world(c, *frame) @ np.linalg.inv(M) + center for c in corners])
    res.shape
    return res


def get_rot_mat(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array([
        [c, -s],
        [s, c]
    ])
    return rot

def rewarp_image(img, from_centroid, frame, to_centroid, to_size, mask, M, downsample, premultiply=True):
    if premultiply:
        if img.shape != mask.shape:
            img *= mask[..., np.newaxis]
        else:
            img *= mask
    # plt.imshow(mask); plt.colorbar(); plt.show()

    # Move to world space
    mat1 = registration.get_warp(frame[0], frame[1], [0, 0])
    # Translate by gps coordinates
    mat2 = registration.get_warp([0, 0], np.identity(2), -(to_centroid - from_centroid) @ M)
    # Downsample
    mat2[:-1] /= downsample

    # Move to camera space of other image (in two parts))
    mat3 = registration.get_warp([0, 0], np.identity(2), -to_size[:2][::-1] / 2)
    
    mat = mat3 @ mat2 @ mat1
    
    img_shape = mask.shape
    warped = cv2.warpAffine(
        img,
        mat[:-1],
        to_size[:2][::-1]
    )
    
    mask_out = cv2.warpAffine(
        mask,
        mat[:-1],
        to_size[:2][::-1]
    )
    
    return warped, mask_out

def mask_pyramid(mask, n):
    if n <= 0:
        return mask, (mask >= 1).astype(np.float32)
    recur, score = mask_pyramid(cv2.pyrDown(mask), n - 1)
    recur = cv2.pyrUp(recur.astype(np.float32))[:mask.shape[0], :mask.shape[1]]
    return recur, (mask >= 1).astype(np.float32) + 2 * cv2.pyrUp(score)[:mask.shape[0], :mask.shape[1]]

def pyramid_score(img):
    score = 0
    gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 240).astype(np.float32)
    eroded = cv2.erode(gray, np.ones((3, 3)))
    return mask_pyramid(eroded, 3)[1] + 1e-3

def normalize(img):
    M = np.mean(img, axis=(0, 1))
    STD = np.std(img, axis=(0, 1))
    print(M, STD)
    return (img - M) / np.where(STD == 0, 1, STD)

def create_tile(centroid, tile_size, pixel_size, images, frames, centroids, bounds, v_mask, k_mask, downsample, M, calibration):
    acc = np.zeros((*pixel_size, 3))
    C = 0
    size = np.array(acc.shape)
    IDS = []

    thresh = np.linalg.norm(np.array(tile_size) / 2)
    mask = np.linalg.norm(centroids - centroid, axis=-1) < thresh * 1.1
    IDS = list(np.arange(len(centroids))[mask])
    mask = v_mask * k_mask
    # for idx, (image, (c, (b, frame))) in enumerate(zip(images, zip(centroids, zip(bounds, frames)))):
    #     # Check if any of the corners are in the tile
    #     if (np.prod(np.abs(b - centroid) < (tile_size / 2)[np.newaxis, ...], axis=-1) > 0).any():
    #         IDS.append(idx)

    # for idx, c in enumerate(centroids)

    if len(IDS) == 0:
        # logger.warning("NO IMAGES FOUND FOR TILE!")
        return None
    K = int(downsample * 2 + 1)
    for i in tqdm(IDS, "Estimating Tile Contents", leave=False):
        image = images[i]
        img = np.asarray(calibration.open(image)).astype(np.float32)
        c = centroids[i]
        frame = frames[i]
        image_mask = mask * cv2.GaussianBlur(pyramid_score(img), [K + (1 - (K % 2)), K + (1 - (K % 2))], downsample, downsample)
        t, m = rewarp_image(
            # cv2.GaussianBlur(img, [K, K], downsample, downsample),
            img,
            c,
            frame,
            centroid,
            size,
            image_mask,
            # mask * cv2.GaussianBlur(mask, [K // 2 + (1 - ((K // 2) % 2)), K // 2 + (1 - ((K // 2) % 2))], downsample / 2, downsample / 2),
            M,
            downsample
        )
        acc += t
        C += m
    m = C > 0
    if type(m) == bool or (not C.any()):
        return None
    
    # res = np.clip(acc / np.maximum(C, 1e-8)[..., np.newaxis], 0, 255).astype(np.uint8)
    # res = cv2.medianBlur(res, 5).astype(np.float32)
    # # plt.imshow(res / 255); plt.show()
    # res = (res - np.mean(res, axis=(0, 1))) / np.std(res, axis=(0, 1))
    # acc = np.zeros((*pixel_size, 3))
    # C = 0
    

    # # MARK: - Add global correlation factor - need to turn off if masked portion is super small


    # for i in tqdm(IDS, "Refining Tile", leave=False):
    #     image = images[i]
    #     img = np.asarray(calibration.open(image)).astype(np.float32)
    #     MEAN = np.mean(img, axis=(0, 1))
    #     STD = np.mean(img, axis=(0, 1))
    #     c = centroids[i]
    #     frame = frames[i]
    #     K = int(downsample)
    #     K += 1 - ((K // 2) % 2)
    #     # image_mask = mask * cv2.GaussianBlur(pyramid_score(img), [K // 2 + (1 - ((K // 2) % 2)), K // 2 + (1 - ((K // 2) % 2))], downsample / 2, downsample / 2)
    #     image_mask = v_mask * (pyramid_score(img) ** 2)

    #     t, m = rewarp_image(
    #         cv2.GaussianBlur(img, [K, K], downsample, downsample),
    #         c,
    #         frame,
    #         centroid,
    #         size,
    #         image_mask,
    #         # mask * cv2.GaussianBlur(mask, [K // 2 + (1 - ((K // 2) % 2)), K // 2 + (1 - ((K // 2) % 2))], downsample / 2, downsample / 2),
    #         M,
    #         downsample,
    #         premultiply=False
    #     )
    #     corr_score = np.mean(
    #             (t - MEAN) / np.where(STD == 0, 1, STD) * res,
    #             axis=-1
    #     )
    #     f_mask = (m > 0)
    #     S = np.sum(f_mask)
    #     if S == 0:
    #         continue
    #     # plt.imshow(corr_score)
    #     global_score = np.exp(np.sum(corr_score) / max(1, S))
    #     if S < 30:
    #         global_score *= 0

    #     local_score = np.exp(
    #         cv2.GaussianBlur(
    #             corr_score, 
    #             [K, K], 
    #             K
    #         )
    #     ) * (m > 0) * global_score

    #     # local_score = global_score = corr_score


    #     # t_mask = (np.mean(t < 240, axis=-1) > 0).astype(np.float32)
    #     # t_mask[np.invert(t_mask.astype(bool))] += 1e-4
    #     # corr_score *= t_mask

    #     # plt.imshow(corr_score); plt.show()
    #     m *= local_score
    #     C += m
    #     acc += t * m[..., np.newaxis]

    #     # plt.imshow(local_score); plt.show()
    #     # print("A", global_score, "B")
    #     # plt.imshow(C); plt.show()
    #     # plt.imshow(
    #     #     np.clip(acc / np.where(C == 0, 1, C)[..., np.newaxis] / 255, 0, 1)
    #     # ); plt.show()
    # # plt.imshow(np.mean(acc, axis=-1)); plt.show()
    # # plt.imshow(C); plt.show()
    return np.clip(acc / np.maximum(C, 1e-8)[..., np.newaxis], 0, 255).astype(np.uint8)[..., ::-1]


import rasterio
from rasterio.transform import from_bounds
def prep_for_rasterio(img):
        # img: (H, W, C) → (C, H, W)
        img = img.transpose(2, 0, 1)  # channels last → first
        return img

def create_tiles(dest, centroids, tile_ids, tile_size, pixel_size, images, frames, image_centroids, bounds, v_mask, k_mask, downsample, M, calibration, thread_id):
    for centroid, id in zip(tqdm(centroids, f"Thread {thread_id}"), tile_ids):
        tile = create_tile(centroid, tile_size, pixel_size, images, frames, image_centroids, bounds, v_mask, k_mask, downsample, M, calibration)


        if not tile is None:
            MIN = (centroid - tile_size / 2).astype(np.float64) / 3600 # SW corner 
            MAX = (centroid + tile_size / 2).astype(np.float64) / 3600 # NE corner


            transform = from_bounds(
                # west, south, east, north,
                MIN[1], MIN[0], MAX[1], MAX[0],
                tile.shape[1], tile.shape[0]
            )

            fname = str(dest / f"tile_{id}.tif")

            alpha = (np.sum(tile > 0, axis=-1) >= 3).astype(np.uint8) * 255

            # mat = np.vstack([
            #     np.identity(2), [0, 0]
            # ])
            # mat = np.hstack([
            #     mat,
            #     np.hstack([np.array(tile.shape[:2][::-1]) / 2, [1]]).reshape((3, -1))
            # ])

            # mat2 = np.vstack([
            #     M.T / downsample, [0, 0]
            # ])
            # mat2 = np.hstack([
            #     mat2,
            #     np.array([0, 0, 1]).reshape(3, 1)
            # ])
            # mat_t = np.array([
            #     [1, 0, -centroid[0]],
            #     [0, 1, -centroid[1]],
            #     [0, 0, 1]
            # ])
            # mat3 = mat3 @ np.array([
            #     [3600, 0, 0],
            #     [0, 3600, 0],
            #     [0, 0, 1]
            # ])
            # MAT = mat @ mat2 @ mat3
            
            # mat = np.linalg.inv(MAT)
            # mat = np.array([
            #     [0, 1, 0],
            #     [1, 0, 0],
            #     [0, 0, 1]
            # ]) @ mat

            # transform = Affine(
            #     mat[0, 0], mat[0, 1], mat[0, 2],
            #     mat[1, 0], mat[1, 1], mat[1, 2]
            # )
            # transform = from_bounds(

            # )

            with rasterio.open(
                fname,
                "w",
                driver="GTiff",
                height=tile.shape[0],
                width=tile.shape[0],
                count=4,  # number of bands
                dtype=np.uint8,
                crs="EPSG:4326",  # change if needed
                transform=transform,
            ) as dst:
                d = np.vstack([prep_for_rasterio(tile[..., ::-1]), alpha[np.newaxis,]])
                dst.write(d)
            # cv2.imwrite(fname, tile)
        # else:
        #     logger.warning(f"TILE IS NONE: {id} {centroid}")

import matplotlib.pyplot as plt
from pathlib import Path
from calib import Calibration
from threading import Thread

def create_map(
        project: str,
        downsample: int = 4,
        block_width: int = 1024,
        block_height: int = 1024,
        kernel_strength: float = np.pi ** 3,
        threads: int = 4,
        calibration: str | None = None
        ):
    p = Project(project)
    calibration = Calibration(calibration)

    images = p['images']
    v_mask = p['v_mask']
    shape = v_mask.shape
    centers = gps.to_arc_seconds(p['coords'])

    frames = p["frames"]
    if frames is None:
        frames = [registration.get_image_transform(gps.get_gimbal_yaw(str(image)), shape) for image in tqdm(images, "Getting Image Transforms")]
        p.stash({ "frames": frames })

    image_bounds = p["image_bounds"]
    M = p["transform"]
    if image_bounds is None:
        image_bounds = np.array([get_image_corners(image, frame, center, M, shape) for image, (frame, center) in zip(tqdm(images, "Finding bounds"), zip(frames, centers))])
        p.stash({ "image_bounds": image_bounds })
    map_max = np.max(image_bounds, axis=(0, 1))
    map_min = np.min(image_bounds, axis=(0, 1))
    diff = map_max - map_min

    coords_per_block = np.array([block_width, block_height]) @ np.linalg.inv(M) * (2 ** downsample) # Pixels to LAT/LON
    N = np.abs(np.ceil(diff / coords_per_block))

    X2, X1 = np.meshgrid(np.arange(N[1]), np.arange(N[0]))
    X = np.dstack([X1.flatten(), X2.flatten()])[0]

    # Calculating weighting for distance
    v_mask = p['v_mask']
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, v_mask.shape[1]), np.linspace(-0.5, 0.5, v_mask.shape[0]))
    DSQ = u * u + v * v
    d_mask = np.exp(-DSQ * kernel_strength) ** 3

    dest = Path(project).parent / "geotiles"
    if not dest.exists():
        os.mkdir(str(dest))

    np.random.shuffle(X)
    log(f"Creating {X.shape[0]} tiles")
    n_per_thread = int(np.ceil(X.shape[0] / threads))
    threads = []
    for thread_id, i in enumerate(range(0, X.shape[0], n_per_thread)):
        thread = Thread(
                target=create_tiles,
                args=(
                    dest,
                    X[i:min(i + n_per_thread, X.shape[0])] * diff / N + map_min,
                    np.arange(i, min(i + n_per_thread, X.shape[0])),
                    diff / N,
                    (block_width, block_height),
                    images,
                    frames,
                    centers,
                    image_bounds,
                    v_mask,
                    d_mask,
                    2 ** downsample,
                    p['transform'],
                    calibration,
                    thread_id + 1
                )
            )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()



if __name__ == "__main__":
    tyro.cli(create_map)