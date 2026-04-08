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
    if logs:
        log(f"Found {len(lines)} lines")
    line_mask = np.ones_like(mask)
    X, Y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
    
    coords = np.dstack([Y, X])
    if lines is not None:
        for line in tqdm(lines):
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

def find_circles(mask, dp=2, minDist=20, param1=5, param2=20, minRadius=5, maxRadius=30):
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
    img = mask_highlight(img, mask)
    if circles is not None:
        C = np.round(circles[0, :]).astype("int")
        for (x, y, r) in C:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
    return img

        
def find_floats(img, draw_results=True, logs=True):
    if logs:
        log("Finding Gray Mask")
    g_mask, gray = gray_mask(img)
    if logs:
        log("Finding Boat Mask")
    b_mask = area_mask(g_mask)
    if logs:
        log("Finding Edge Mask")
    e_mask = edge_mask(img)

    acc_mask = np.bitwise_and(g_mask, np.bitwise_and(b_mask, e_mask))

    if logs:
        log("Finding Line Mask")
    l_mask = line_mask(acc_mask, logs=logs)
    acc_mask = np.bitwise_and(acc_mask, l_mask)
    if logs:
        log("Finding Circles")
    circles = find_circles(acc_mask)
    res = {
        'circles': circles,
        'g_mask': g_mask,
        'b_mask': b_mask,
        'l_mask': l_mask,
        'acc_mask': acc_mask,
    }
    if draw_results:
        res.update({'result': draw_circles(img, acc_mask, circles)})
    return res