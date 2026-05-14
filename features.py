import cv2, numpy as np, matplotlib.pyplot as plt, glob
from tqdm import tqdm
from loguru import logger
log = logger.debug


def mask_highlight(img, mask, highlight=0.75):
    return img * (mask * highlight + (1 - highlight))[..., np.newaxis]

"""
Creates a mask of white objects
"""
def gray_mask(img, threshold=0.5):
    gray = np.mean(img, axis=-1)
    mask = np.linalg.norm(img / 255 - np.array([1, 1, 1]), axis=-1) < threshold
    return mask, gray

"""
Creates a mask of mostly solid objects
"""
def gradient_mask(img, kernel_size=3, threshold=0.05):
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0) / 255
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1) / 255
    dsq = dx * dx + dy * dy
    d = np.mean(dsq, axis=-1)
    d = cv2.medianBlur(d, kernel_size) < threshold
    d = cv2.erode(d.astype(np.uint8), np.ones((kernel_size, kernel_size)))
    d = cv2.dilate(d, np.ones((kernel_size + 2, kernel_size + 2)))
    return d

"""
Removes large areas in a mask
"""
def area_mask(gray_mask, downsamples=2, median_size=15, dilation_size=11):
    mask = gray_mask.copy().astype(np.uint8)
    for i in range(downsamples):
        mask = cv2.pyrDown(mask)

    mask = cv2.dilate(
        cv2.medianBlur(mask, median_size),
        np.ones((dilation_size, dilation_size))
    )
    
    for i in range(downsamples):
        mask = cv2.pyrUp(mask)
    mask = mask[:gray_mask.shape[0], :gray_mask.shape[1]]
    mask = 1 - mask
    return mask

""" Mask Areas around edges
This should be sensitive, as floats will also have edges!
"""
def edge_mask(img, blur_size=3, canny1=100, cann2=100):
    return np.invert(
        cv2.GaussianBlur(
            cv2.Canny(img, canny1, cann2),
            [blur_size, blur_size], 
            blur_size
        ) > 0
    )

"""Removes Lines
dist_thresh: distance from a line before signal is kept

Look at cv2.HoughLinesP for parameters
"""
def line_mask(mask, dist_thresh=10, rho=1, theta=np.pi / 180, threshold=10, minLineLength=50, maxLineGap=10, logs=True):
    lines = cv2.HoughLinesP(
        mask,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=minLineLength,
        maxLineGap=maxLineGap
    )
    if logs and not lines is None:
        log(f"Found {len(lines)} lines")
    if lines is None:
        return np.ones_like(mask)
    line_mask = np.ones_like(mask)
    X, Y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
    
    coords = np.dstack([Y, X])
    if lines is not None:
        for line in tqdm(lines, "Removing Lines", leave=False):
            x1, y1, x2, y2 = line[0]
            p1 = np.array([y1, x1]).astype(np.float32)
            p2 = np.array([y2, x2]).astype(np.float32)
            v = p2 - p1
            L = np.linalg.norm(v)
            v /= L
            V = coords - p1
            t = np.vecdot(V, v)
            P = np.linalg.norm(V - v * t[..., np.newaxis], axis=-1)
            M = np.invert(np.bitwise_and(P < dist_thresh, np.bitwise_and(t > dist_thresh, t < L + dist_thresh)))
            line_mask = np.bitwise_and(line_mask, M)
    return line_mask

def find_circles(mask, dp=2, minDist=500, param1=5, param2=20, minRadius=5, maxRadius=30):
    circles = cv2.HoughCircles(
        mask.astype(np.uint8),
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,   # edge detection threshold
        param2=param2,    # circle detection sensitivity
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    return circles

def draw_circles(img, mask, circles):
    if not mask is None:
        img = mask_highlight(img, mask)
    else:
        img = img.copy()
    if circles is not None:
        C = np.round(circles[0, :]).astype("int")
        for (x, y, r) in C:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
    return img

from scipy import ndimage     
def find_floats_direct(img, gray=0.15, erosion=3, dilation=5, g_mask=0.05, draw_results=True, logs=True):
    if logs:
        log("Finding Gray Mask")
    mask, gray = gray_mask(img, threshold=gray)
    if logs:
        log("Finding Boat Mask")
    b_mask = area_mask(mask)
    acc_mask = np.bitwise_and(mask, b_mask)
    if logs:
        log("Finding Edge Mask")
    e_mask = edge_mask(img, canny1=500, cann2=0)
    acc_mask = np.bitwise_and(acc_mask, e_mask)

    if g_mask:
        if logs:
            log("Taking Gradient Mask")
        g_mask = gradient_mask(img, threshold=g_mask)
        acc_mask = np.bitwise_and(acc_mask, g_mask)

    res = {}

    if logs:
        log("Finding Line Mask")
    l_mask = line_mask(acc_mask, logs=logs)
    acc_mask = np.bitwise_and(acc_mask, l_mask)
    acc_mask = cv2.erode(
        acc_mask,
        np.ones((erosion, erosion))
    )
    acc_mask = cv2.dilate(
        acc_mask,
        np.ones((dilation, dilation))
    )

    if logs:
        log("Finding Circles")
    circles = find_circles(acc_mask)
    res.update({
        'circles': circles,
        'acc_mask': acc_mask,
    })
    if logs:
        res.update(
            {
                'mask': mask,
                'g_mask': g_mask,
                'b_mask': b_mask,
                'e_mask': e_mask,
                'l_mask': l_mask,
            }
        )
    if draw_results:
        res.update({'result': draw_circles(img, acc_mask, circles)})
    return res

def find_floats_direct(img, gray=0.15, erosion=3, dilation=5, g_mask=0.05, draw_results=True, logs=True):
    if logs:
        log("Finding Gray Mask")
    mask, gray = gray_mask(img, threshold=gray)
    if logs:
        log("Finding Boat Mask")
    b_mask = area_mask(mask)
    acc_mask = np.bitwise_and(mask, b_mask)
    if logs:
        log("Finding Edge Mask")
    e_mask = edge_mask(img, canny1=500, cann2=0)
    acc_mask = np.bitwise_and(acc_mask, e_mask)

    if g_mask:
        if logs:
            log("Taking Gradient Mask")
        g_mask = gradient_mask(img, threshold=g_mask)
        acc_mask = np.bitwise_and(acc_mask, g_mask)

    res = {}

    if logs:
        log("Finding Line Mask")
    l_mask = line_mask(acc_mask, logs=logs)
    acc_mask = np.bitwise_and(acc_mask, l_mask)
    acc_mask = cv2.erode(
        acc_mask,
        np.ones((erosion, erosion))
    )
    acc_mask = cv2.dilate(
        acc_mask,
        np.ones((dilation, dilation))
    )

    if logs:
        log("Finding Circles")
    circles = find_circles(acc_mask)
    res.update({
        'circles': circles,
        'acc_mask': acc_mask,
    })
    if logs:
        res.update(
            {
                'mask': mask,
                'g_mask': g_mask,
                'b_mask': b_mask,
                'e_mask': e_mask,
                'l_mask': l_mask,
            }
        )
    if draw_results:
        res.update({'result': draw_circles(img, acc_mask, circles)})
    return res

def estimate_ellipses(labeled_mask, num_features, feature_mask):
    # Find edges
    edges = cv2.Canny(feature_mask.astype(np.uint8), 3, 3, 3, 3)
    
    # Inflate signals for edges
    IDS = ndimage.maximum_filter(labeled_mask, size=5)

    M = np.zeros((num_features, 2), dtype=np.float32)
    ESQ = np.zeros((num_features, 2), dtype=np.float32)
    EXY = np.zeros(num_features, dtype=np.float32)
    N = np.zeros(num_features, dtype=np.float32)
    coords = np.dstack((np.meshgrid(np.arange(feature_mask.shape[1]), np.arange(feature_mask.shape[0])))[::-1])

    edge_mask = edges > 0 
    # edge_mask[edge_mask] *= IDS[edge_mask] > 0
    edge_id = IDS[edge_mask]

    C = coords[edge_mask]
    np.add.at(M, edge_id, C)
    np.add.at(ESQ, edge_id, C * C)
    np.add.at(EXY, edge_id, np.prod(C, axis=-1))
    np.add.at(N, edge_id, 1)
    N = np.maximum(N, 1)
    
    M /= N[..., np.newaxis]
    ESQ /= N[..., np.newaxis]
    EXY /= N
    MSQ = ESQ - M * M

    cross = EXY - np.prod(M, axis=-1)
    diff = MSQ[:, 0] - MSQ[:, 1]
    det = np.sqrt(diff * diff + 4 * cross * cross)

    s = np.sum(MSQ, axis=-1)
    A = np.sqrt(s + det)

    # B = np.sqrt(s - det)
    eccentricity = np.where(s > det, np.sqrt(s - det) / A, 0)
    
    orientation = -0.5 * np.atan2(
        2 * cross,
        MSQ[:, 0] - MSQ[:, 1]
    ) + np.pi / 2
    
    return eccentricity, orientation

def find_floats(img, gray=220, erosion=3, dilation=5, g_mask=0.05, size_constraints=(150, 600), eccentricity_threshold=0.075, draw_results=True, logs=True):
    if logs:
        log("Finding Gray Mask")
    mask = (np.mean(img, axis=-1) > gray).astype(np.float32)

    labeled_mask, num_features = ndimage.label(mask)

    areas = ndimage.sum(mask, labeled_mask, index=range(1, num_features + 1))
    labeled_mask -= 1

    THRESH = size_constraints[0]
    THRESH2 = size_constraints[1]
    IDS = np.where(labeled_mask >= 0, areas[labeled_mask], 0)
    feature_mask = (np.bitwise_and(IDS > THRESH, IDS < THRESH2))
    eccentricity, orientation = estimate_ellipses(labeled_mask, num_features, feature_mask.astype(np.uint8))
    feature_mask = (np.square(np.where(labeled_mask > 0, eccentricity[labeled_mask], 0) - 1) < eccentricity_threshold) * feature_mask
    res = {}

    if logs:
        res.update({"feature_mask": feature_mask})
        log("Finding Edge Mask")
    e_mask = edge_mask(img, canny1=500, cann2=0)
    acc_mask = np.bitwise_and(feature_mask, e_mask)

    if g_mask:
        if logs:
            log("Taking Gradient Mask")
        g_mask = gradient_mask(img, threshold=g_mask)
        acc_mask = np.bitwise_and(acc_mask, g_mask)

    if logs:
        log("Finding Line Mask")
    l_mask = line_mask(acc_mask, logs=logs)
    acc_mask = np.bitwise_and(acc_mask, l_mask)
    acc_mask = cv2.erode(
        acc_mask,
        np.ones((erosion, erosion))
    )
    acc_mask = cv2.dilate(
        acc_mask,
        np.ones((dilation, dilation))
    )

    if logs:
        log("Finding Circles")
    circles = find_circles(acc_mask)
    res.update({
        'circles': circles,
        'acc_mask': acc_mask,
    })
    if logs:
        res.update(
            {
                'mask': mask,
                'g_mask': g_mask,
                'e_mask': e_mask,
                'l_mask': l_mask,
            }
        )
    if draw_results:
        res.update({'result': draw_circles(img, acc_mask, circles)})
    return res