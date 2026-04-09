import registration
import gps
import numpy as np
import cv2
from PIL import Image

def rewarp_image(from_path, to_path, img_shape, v_mask, M):
    c1 = registration.get_image_transform(gps.get_gimbal_yaw(from_path), img_shape)
    c2 = registration.get_image_transform(gps.get_gimbal_yaw(to_path), img_shape)
    T = gps.to_arc_seconds(gps.get_coords(to_path) - gps.get_coords(from_path))
    
    # Move to world space
    mat1 = registration.get_warp(c1[0], c1[1], [0, 0])
    # Translate by gps coordinates
    mat2 = registration.get_warp([0, 0], np.identity(2), -T @ M)
    # Move to camera space of other image (in two parts)
    mat3 = registration.get_warp([0, 0], c2[1].T, [0, 0])
    mat4 = registration.get_warp([0, 0], np.identity(2), -c2[0])
    
    mat = mat4 @ mat3 @ mat2 @ mat1
    
    warped = cv2.warpAffine(
        np.asarray(Image.open(from_path)) * v_mask[..., np.newaxis],
        mat[:-1],
        img_shape[:2][::-1]
    )
    
    mask = cv2.warpAffine(
        v_mask,
        mat[:-1],
        img_shape[:2][::-1]
    )
    
    return warped, mask

def rewarp(safe_transforms, arc_coords, query_id, imgs, M):
    Q = query_id
    
    ID1 = safe_transforms[Q][0][0]
    ID2 = safe_transforms[Q][0][1]
    transform = safe_transforms[Q]
    
    c1 = registration.get_image_transform(gps.get_gimbal_yaw(targets[ID1]), imgs[0].shape)
    c2 = registration.get_image_transform(gps.get_gimbal_yaw(targets[ID2]), imgs[0].shape)
    
    T = arc_coords[ID2] - arc_coords[ID1]
    
    # Move to world space
    mat1 = registration.get_warp(c1[0], c1[1], [0, 0])
    # Translate by gps coordinates
    mat2 = registration.get_warp([0, 0], np.identity(2), -T @ M)
    # Move to camera space of other image (in two parts)
    mat3 = registration.get_warp([0, 0], c2[1].T, [0, 0])
    mat4 = registration.get_warp([0, 0], np.identity(2), -c2[0])
    
    mat = mat4 @ mat3 @ mat2 @ mat1
    
    warped = cv2.warpAffine(
        np.mean(imgs[ID1], axis=-1),
        mat[:-1],
        img_shape[:2][::-1]
    )
    return np.dstack([
        warped,
        np.mean(imgs[ID2], axis=-1),
        np.zeros_like(warped)
    ])