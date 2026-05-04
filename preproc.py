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

from project import Project

def get_target_area(images, target, target_area):
    target_coord = gps.get_coords(target)

    img_exif = [(im, gps.get_exif_data(im))for im in tqdm(images, "Loading EXIF Data")]
    coords = np.array([gps.get_coords(i[1]) for i in tqdm(img_exif, "Extracing GPS Data")])
    
    mask = gps.dist_from_centroid(coords, target_coord) < target_area # Use only images within 2 arcseconds of the target
    
    targets = np.array(images)[mask]
    # mask2 = np.array([np.abs(gps.get_gimbal_pitch(t) + 90) < 0.3 for t in targets])
    # log(f"Rejecting {np.sum(np.invert(mask2))} target images due to unsuitable gimball angles")
    
    # mask[mask] = mask2
    # targets = targets[mask2]
    log(f"{len(targets)} target images")
    return targets, mask, coords
    

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
    plt.close()
    return imgs

def variance(images, n):
    e = 0
    esq = 0
    for i in tqdm(np.random.randint(0, len(images), n), "Estimating Variance for Vignette Mask"):
        im = np.asarray(Image.open(images[i])).astype(np.float32)
        e += im
        esq += im * im
        
    variance = np.linalg.norm(e * e - esq, axis=-1)
    return variance

def get_v_mask(images, n, edge_erosion, threshold=1/4, kernel_size=105):
    v = variance(images, n)
    v_mask = cv2.erode((v > v.mean() * threshold).astype(np.uint8), np.ones((kernel_size, kernel_size)))
    v_mask[:edge_erosion] = 0
    v_mask[-edge_erosion:] = 0
    v_mask[:, :edge_erosion] = 0
    v_mask[:, -edge_erosion:] = 0
    return v_mask

def draw_features(imgs, float_results, working):
    features_dir = working / "features"
    if not features_dir.exists():
        os.mkdir(str(features_dir))
    for idx, (img, results) in enumerate(zip(tqdm(imgs, "Drawing Target Results"), float_results)):
        fig, ax = plt.subplots(1, figsize=np.array(imgs[0].shape[:2][::-1]) / np.max(imgs[0].shape[::-1]) * 4)
        I = features.draw_circles(
            img.copy(),
            results['acc_mask'],
            results['circles']
        )
        ax.imshow(I / 255)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(str(features_dir / f"features_{idx}.jpg"), dpi=300)
        plt.close()

def visualize_mapping_error(targets, target_coords, shape, v_mask, M, destination, id=None):
    errors = np.sum(np.linalg.norm(target_coords[:, np.newaxis] - target_coords, axis=-1), axis=-1)
    target_id = np.argmin(errors)
    target = targets[target_id]

    R = 0
    C = 0
    for t in tqdm(targets, "Rewarping Images to Target"):
        res, mask = warping.rewarp_image(str(t), str(target), shape, v_mask, M)
        R += res.astype(np.float32)
        C += mask
    
    result = np.clip(R / np.maximum(C, 1)[..., np.newaxis], 0, 255).astype(np.uint8)
    plt.imshow(
        result
    )
    if not destination is None:
        plt.savefig(str(destination / f"mapping_spread{"" if id is None else id}.jpg"), dpi=300)
        plt.close()
    
def get_good_images(data_root):
    log("Finding images")
    images = sorted(glob.glob(str(Path(data_root) / "*/*.JPG")))
    if len(images) == 0:
        images = sorted(glob.glob(str(Path(data_root) / "*.JPG")))

    log("Removing bad images")
    mask = np.array([np.abs(gps.get_gimbal_pitch(t) + 90) < 0.3 for t in tqdm(images, "Finding Gimbal Pitches")])
    log(f"Rejecting {np.sum(np.invert(mask))} target images due to unsuitable gimball angles")
    images = [im for im, m, in zip(images, mask) if m]
    return images


def redo_transform(p):
    potentials = []
    scores = []

    N = 4
    for _ in range(30):
        idxs = np.random.randint(0, p['offsets'].shape[0], N)
        o = p['offsets'][idxs]
        t = p['translations'][idxs]
        M = np.linalg.lstsq(o, t)[0]
        potentials.append(M)
        scores.append(np.mean(np.linalg.norm(p['offsets'] @ M - p['translations'], axis=-1) < 50))
        
    best = potentials[np.argmax(scores)]
    log(f"Redoing transform– Scores: {np.max(scores)}")
    return best

from registration import TransformMode

def main(
    data: str,
    working_directory: str, 
    target: str, 
    save_masks: bool = False,
    draw_matches: bool = True,
    target_area: float = 3,
    prune_features: bool = True,
    transform_mode: TransformMode = TransformMode.Rectified
):
    log("Initializing: setting up directories")
    if not Path(working_directory).exists():
        os.mkdir(working_directory)
    working = Path(working_directory)

    project_path = Path(working / "project.npy")
    project = Project(project_path)
    
    if project["images"] is None:
        images = get_good_images(data)

        assert(len(images) > 0)
        log(f"Working with {len(images)} images.")

        targets, mask, coords = get_target_area(images, target, target_area)
        imgs = summarize_data(coords, working, targets, mask)

        project.initialize_dataset(images, targets, coords)
    else:
        imgs = None
        images = project["images"]; targets = project["targets"]; coords = project["coords"]

    if project["features"] is None:
        if imgs is None:
            imgs = [np.asarray(Image.open(t)) for t in tqdm(project["targets"], "Loading Images")]
        float_results = [features.find_floats(im, logs=save_masks) for im in tqdm(imgs, "Finding Features")]
        circles = [i['circles'][0] if not i['circles'] is None else np.array([]).reshape((-1, 3)) for i in float_results]
        coordinate_frames = [registration.get_image_transform(gps.get_gimbal_yaw(t), imgs[0].shape) for t in targets]
        draw_features(imgs, float_results, working)
        project.save_features((float_results, circles, coordinate_frames))
    else:
        float_results, circles, coordinate_frames = project["features"]


    match_dir = None
    if draw_matches:
        match_dir = working / "matching"
        if not match_dir.exists():
            os.mkdir(str(match_dir))

    # Match Floats
    if project["matches"] is None:
        safe_transforms, transforms = registration.get_good_transforms(targets, circles, min_correspondances=3, desc_size=100, match_dir=match_dir)
        project.save_matches((safe_transforms, transforms))
    else:
        safe_transforms, transforms = project["matches"]

    
    # Get a conversion from GPS coordinates to image coordinates
    target_coords = np.array([gps.get_coords(str(t)) for t in tqdm(targets)])
    
    v_mask = project["v_mask"]
    if project["v_mask"] is None:
        v_mask = get_v_mask(images, 200, 30)
        project.save_vignette_mask(v_mask)
    shape = v_mask.shape

    if project["transform"] is None:
        M, offsets, translations = registration.map_coords_to_image(targets, safe_transforms, gps.to_arc_seconds(target_coords))
        project.save_triangulation(M, offsets, translations)
        visualize_mapping_error(targets, target_coords, shape, v_mask, M, working)
        # best = redo_transform(project)
        # M = best
        # project['transform'] = best
        # project.write()
        # visualize_mapping_error(targets, target_coords, shape, v_mask, M, working, id="_REMAP")
    else:
        M = project["transform"]
        offsets = project["offsets"]
        transforms = project["translations"]

    if prune_features:
        r = project['features']
        r = (None, r[1], r[2])
        project['features'] = r

    project.write()


if __name__ == "__main__":
    tyro.cli(main)