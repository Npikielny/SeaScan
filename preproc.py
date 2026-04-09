import cv2, numpy as np, matplotlib.pyplot as plt, glob
from tqdm import tqdm
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import warping

import gps, features, registration

from loguru import logger
import tyro

log = logger.debug

from pathlib import Path
import os

import project

def get_target_area(images, coords, target, target_area):
    target_coord = gps.get_coords(target)
    
    mask = gps.dist_from_centroid(coords, target_coord) < target_area # Use only images within 2 arcseconds of the target
    
    targets = np.array(images)[mask]
    mask2 = np.array([np.abs(gps.get_gimbal_pitch(t) + 90) < 0.3 for t in targets])
    log(f"Rejecting {np.sum(np.invert(mask2))} target images due to unsuitable gimball angles")
    
    mask[mask] = mask2
    targets = targets[mask2]
    log(f"{len(targets)} target images")
    return targets, mask
    

def summarize_data(coords, working, targets, mask):
    LATLON = gps.to_arc_seconds(coords)
    imgs = [
        np.asarray(Image.open(im)) for im in tqdm(targets, "Loading Targets for Visualization")
    ]
    
    fig, axs = plt.subplots(2, figsize=(np.array([len(imgs), len(imgs)]) * np.array([imgs[0].shape[:2][::-1]]) / np.max(imgs[0].shape) * 2)[0])
    
    c = axs[0].scatter(
        LATLON[:, 0],
        LATLON[:, 1],
        c=np.arange(LATLON.shape[0]) / LATLON.shape[0],
        # s=2
        cmap='jet'
    )
    fig.colorbar(c)
    
    axs[0].scatter(
        LATLON[mask][:, 0],
        LATLON[mask][:, 1],
        s=1,
        c='white'
    )
    axs[0].set_aspect('equal')
    
    axs[0].axis('off') # The axis is sometimes anisotropically scaled, which is confusing, so I turned off labels
    
    axs[1].imshow(np.hstack(imgs))
    _ = axs[1].axis('off')
    fig.savefig(Path(working / "data_summary.jpg"), dpi=300)
    return imgs

def variance(images, n):
    e = 0
    esq = 0
    for i in tqdm(np.random.randint(0, len(images), n)):
        im = np.asarray(Image.open(images[i])).astype(np.float32)
        e += im
        esq += im * im
        
    variance = np.linalg.norm(e * e - esq, axis=-1)
    return variance

def draw_features(imgs, float_results, working):
    width = 4
    height = int(np.ceil(len(imgs) / width))
    
    fig, axs = plt.subplots(height, width, figsize=[width, height] * np.array(imgs[0].shape[:2][::-1]) / np.max(imgs[0].shape[::-1]) * 12)
    for ax, (img, results) in zip(axs.flatten(), zip(imgs, tqdm(float_results, "Drawing Target Results"))):
        I = features.draw_circles(
            img.copy(),
            results['acc_mask'],
            results['circles']
        )
            
        ax.imshow(I / 255)
    for ax in axs.flatten():    
        ax.axis('off')
        
    fig.tight_layout()
    _ = fig.savefig(str(working / 'features.jpg'), dpi=300)

def visualize_mapping_error(targets, target, shape, v_mask, M):
    R = 0
    C = 0
    for t in tqdm(targets):
        res, mask = warping.rewarp_image(str(t), target, shape, v_mask, M)
        R += res.astype(np.float32)
        C += mask
    
    result = np.clip(R / np.maximum(C, 1)[..., np.newaxis], 0, 255).astype(np.uint8)
    plt.imshow(
        result
    )
    plt.savefig("mapping_spread.jpg", dpi=300)
    
def main(
    data: str,
    working_directory: str, 
    target: str, 
    draw_matches: bool = True,
    target_area: float = 3
):
    images = sorted(glob.glob(str(Path(data) / "*/*.JPG")))
    os.mkdir(working_directory)
    working = Path(working_directory)
    assert(len(images) > 0)
    log(f"Working with {len(images)} images.")
    
    img_exif = [(im, gps.get_exif_data(im))for im in tqdm(images, "Loading EXIF Data")]
    coords = np.array([gps.get_coords(i[1]) for i in tqdm(img_exif, "Extracing GPS Data")])

    targets, mask = get_target_area(images, coords, target, target_area)
    imgs = summarize_data(coords, working, targets, mask)

    float_results = [features.find_floats(im, logs=True) for im in tqdm(imgs)]
    
    circles = [i['circles'][0] for i in float_results]
    coordinate_frames = [registration.get_image_transform(gps.get_gimbal_yaw(t), imgs[0].shape) for t in targets]
    draw_features(imgs, float_results, working)

    match_dir = None
    if draw_matches:
        match_dir = working / "matching"
        os.mkdir(str(match_dir))
    # Match Floats
    safe_transforms, transforms = registration.get_good_transforms(targets, circles, min_correspondances=3, desc_size=100, match_dir=match_dir)
    
    # Get a conversion from GPS coordinates to image coordinates
    target_coords = np.array([gps.get_coords(str(t)) for t in tqdm(targets)])
    
    M, offsets, translations = registration.map_coords_to_image(targets, safe_transforms, gps.to_arc_seconds(target_coords))

    v = variance(images, 200)
    v_mask = cv2.erode((v > v.mean() / 4).astype(np.uint8), np.ones((105, 105)))
    shape = imgs[0].shape
    visualize_mapping_error(targets, target, shape, v_mask, M)

    project.write_project(
        working / "project.npy", 
        target, 
        images,
        M, 
        v_mask
    )



    

    

if __name__ == "__main__":
    tyro.cli(main)