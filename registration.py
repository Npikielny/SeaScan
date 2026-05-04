import cv2, numpy as np, matplotlib.pyplot as plt, glob
from tqdm import tqdm
from loguru import logger
log = logger.debug
from pathlib import Path

def translate_to(t):
    return np.array([
        [1, 0, -t[0]],
        [0, 1, -t[1]],
        [0, 0, 1]
    ]).astype(float)

def get_warp(ctr, rot, offset):
    # Move to center
    # Rotate
    # Move back
    T = translate_to(ctr)
    return translate_to(offset) @ np.vstack([
    np.hstack([rot, [[0], [0]]]),
        [0, 0, 1]
    ]) @ T

def create_descriptors(image, centers, descriptor_radius, coordinate_frame=None):
    img = image.astype(np.float32)
    if coordinate_frame is None:
        rot = np.identity(2)
    else:
        rot = coordinate_frame[1]

    descriptors = []
    for c in tqdm(centers, "Creating Image Patches", leave=False):
        warp = get_warp(c, rot, [-descriptor_radius, -descriptor_radius])
        descriptors.append(
            cv2.warpAffine(
                image, 
                warp[:-1],
                [descriptor_radius * 2 + 1, descriptor_radius * 2 + 1],
                borderMode=cv2.BORDER_REPLICATE
            )
        )
    if len(descriptors) > 0:
        means = np.mean(descriptors, axis=(1, 2))
        std = np.std(descriptors, axis=(1,2))
        descriptors -= means[:, np.newaxis, np.newaxis, :]
        descriptors /= std[:, np.newaxis, np.newaxis, :]
        return np.linalg.norm(descriptors, axis=-1)
    return []
    
def find_best_score(desc, options):
    best = 0
    best_score = -np.inf
    for i, o in enumerate(options):
        s = np.sum(o * desc)
        # res = cv2.matchTemplate(desc[i], o, cv2.TM_CCOEFF)
        # _, s, _, max_loc = cv2.minMaxLoc(res)

        if s > best_score:
            best = i
            best_score = s
    return (best, best_score)

def find_worst_duplicates(scores):
    res = {}
    for i in range(scores.shape[0]):
        r = res.get(scores[i, 0], [])
        r.append((i, scores[i, 1]))
        res[scores[i, 0]] = r

    worst_dup = []
    for i in res:
        if len(res[i]) > 1:
            best = np.argmax(np.array(res[i])[:, 1])
            for j in range(len(res[i])):
                if j == best:
                    continue
                worst_dup.append(res[i][j][0])
    return worst_dup

def match_descriptors(desc1s, desc2s):
    if len(desc1s) == 0 or len(desc2s) == 0:
        logger.warning("IMAGE WITH 0 DESCRIPTORS, SKIPPING MATCH")
        return np.zeros(len(desc1s)), np.zeros(len(desc1s))

    scores = np.array([find_best_score(d, desc2s) for d in desc1s])
    while len(np.unique(scores[:, 0])) != len(desc2s) and len(np.unique(scores[:, 0])) != len(desc1s):
        # print("REMOVING DOUBLES", len(np.unique(scores[:, 0])), len(desc2s), len(desc1s))
        remaining = np.arange(len(desc2s))#.shape[0])
        remaining = remaining[np.invert(np.isin(remaining, scores[:, 0]))]
        checking = find_worst_duplicates(scores)
        s = np.array([find_best_score(d, desc2s[remaining]) for d in desc1s[checking]])
        s[:, 0] = np.array(remaining)[s[:, 0].astype(int)]
        scores[checking] = s

    if len(np.unique(scores[:, 0])) != len(desc1s):
        assert(len(desc2s) < len(desc1s))
        dups = find_worst_duplicates(scores)
        mask = np.invert(np.isin(np.arange(len(desc1s)), dups))
        return scores[mask], mask
    else:
        return scores, np.ones(len(desc1s))

from PIL import Image
import gps
def match_images(paths, circles, matching_query=None, DESC_SIZE=60):
    descriptors = []
    frames = []
    for path, res in zip(paths, tqdm(circles, "Creating Descriptors")):
        image = np.asarray(Image.open(path))
        yaw = gps.get_gimbal_yaw(path)
        frame = get_image_transform(yaw, image.shape)
        frames.append(frame)
        centers = (np.round(res[:, :2]))#.astype(int)
        desc = create_descriptors(image, centers.astype(int), DESC_SIZE, frame)

        c0 = centers
        centers = (centers - frame[0]) @ frame[1].T + frame[0]
        descriptors.append(
            (
                centers,
                desc, 
                c0
            )
        )

    if matching_query is None:
        matching_query = []
        for i in range(len(paths)):
            for j in range(len(paths)):
                if i == j:
                    continue
                matching_query.append((i, j))

    results = [match_descriptors(descriptors[a][1], descriptors[b][1]) for (a, b) in tqdm(matching_query, "Matching Queries")]
    return descriptors, results, frames

        
def draw_matches(path1, path2, matches, centers1, centers2):
    im1 = np.asarray(Image.open(path1))
    im2 = np.asarray(Image.open(path2))
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = im1
    canvas[:h2, w1:w1 + w2] = im2
    
    if len(matches) > 0 and len(matches.shape) > 1:
        c1 = centers1 
        c2 = centers2[matches[:, 0].astype(int)]
        
        # Draw matches
        for (pt1, pt2) in zip(c1, c2):
            # pt1 = (int(x1), int(y1))
            # pt2 = (int(x2 + w1), int(y2))  # offset x2 by width of img1
        
            # Draw circles
            pt2 += [w1, 0]
            cv2.circle(canvas, pt1, 5, (0, 255, 0), -1)
            cv2.circle(canvas, pt2, 5, (0, 255, 0), -1)
        
            # Draw connecting line
            cv2.line(canvas, pt1, pt2, (255, 0, 0), 2)
    else:
        try:
            logger.warning(f"Insufficient matches or mishapen matches: {matches.shape}")
        except:
            logger.warning(f"Insufficient matches or mishapen matches: {len(matches)}")

    return canvas

def ransac_t(c1, c2, pct_error=0.01):
    t = c2 - c1
    best = 0
    best_count = -np.inf
    best_score = np.inf
    for idx, i in enumerate(t):
        score = np.abs((t @ i) / (i @ i) - 1)
        mask = score < pct_error
        s = np.sum(mask)
        score = np.mean(score[mask])
        # print(s, score)
        if s > best_count or s == best_count and score < best_score:
            best = idx
            best_count = s
            best_score = score

    score = np.abs((t @ t[best]) / (t[best] @ t[best]) - 1)
    mask = score < pct_error
    return t[best], best, best_score, best_count, mask


def ransac_d(c1, c2, error=30):
    t = c2 - c1
    best = 0
    best_count = -np.inf
    best_score = np.inf
    for idx, i in enumerate(t):
        score = np.linalg.norm(t - i, axis=-1)
        # score = (t @ i) / (i @ i)
        mask = score < error
        s = np.sum(mask)
        score = np.mean(score[mask])
        if s > best_count or s == best_count and score < best_score:
            best = idx
            best_count = s
            best_score = score

    score = np.linalg.norm(t - t[best], axis=-1)
    mask = score < error
    return t[best], best, best_score, best_count, mask

def get_image_transform(gimball_yaw, image_shape):
    ctr = np.array(image_shape[:2][::-1]) / 2
    gimball_yaw = gimball_yaw / 180 * np.pi 
    c = np.cos(gimball_yaw)
    s = np.sin(gimball_yaw)
    rot = np.array([
        [c, -s],
        [s, c]
    ])
    return ctr, rot # Convert to inverse

class FeatureMatcher(object):
    def __init__(self):
        pass

    def create_descriptor(self, **kwwargs):
        pass

    """
    Given some descriptor [desc] and a set of descriptors [options], find the best match for [desc]
    """
    def find_best_score(self, desc, options):
        pass

    """
    Given a set of scores: [[ID, score]], find all matches with nonunique IDs that are not the strongest match for that ID
    """
    def find_worst_duplicates(self, scores):
        res = {}
        for i in range(scores.shape[0]):
            r = res.get(scores[i, 0], [])
            r.append((i, scores[i, 1]))
            res[scores[i, 0]] = r

        worst_dup = []
        for i in res:
            if len(res[i]) > 1:
                best = np.argmin(np.array(res[i])[:, 1])
                for j in range(len(res[i])):
                    if j == best:
                        continue
                    worst_dup.append(res[i][j][0])
        return worst_dup
    
    """
    Match the sets of descriptors
    """
    def match_descriptors(self, desc1s, desc2s):
        if len(desc1s) == 0 or len(desc2s) == 0:
            logger.warning("IMAGE WITH 0 DESCRIPTORS, SKIPPING MATCH")
            return np.zeros(len(desc1s)), np.zeros(len(desc1s))

        scores = np.array([self.find_best_score(d, desc2s) for d in desc1s])
        while len(np.unique(scores[:, 0])) != len(desc2s) and len(np.unique(scores[:, 0])) != len(desc1s):
            remaining = np.arange(len(desc2s))#.shape[0])
            remaining = remaining[np.invert(np.isin(remaining, scores[:, 0]))]
            checking = self.find_worst_duplicates(scores)
            s = np.array([self.find_best_score(d, desc2s[remaining]) for d in desc1s[checking]])
            s[:, 0] = np.array(remaining)[s[:, 0].astype(int)]
            scores[checking] = s

        if len(np.unique(scores[:, 0])) != len(desc1s):
            assert(len(desc2s) < len(desc1s))
            dups = self.find_worst_duplicates(scores)
            mask = np.invert(np.isin(np.arange(len(desc1s)), dups))
            return scores[mask], mask
        else:
            return scores, np.ones(len(desc1s))
        
# class PatchMatcher(object):
#     pass

class AgreementMatcher(object):
    def __init__(self, max_translation):
        super().__init__()
        self.max_t = max_translation

    def create_descriptors(self, **kwargs):
        return kwargs.get("positions")

def image_to_world(coord, ctr, rot):
    return (coord - ctr) @ rot # Right multiply is inverse

def world_to_image(coord, ctr, rot):
    return coord @ rot.T + ctr

def get_good_transforms(targets, circles, query_type='exhaustive', ransac_fn=ransac_d, min_correspondances=2, desc_size=100, match_dir=None):
    queries = []
    if query_type == 'exhaustive':
        for i in range(len(targets)):
            for j in range(len(targets)):
                if i >= j: 
                    continue
                queries.append((i, j))
    else:
        query_type = f"Originally {query_type}–switched to sequential"
        for i in range(len(targets) - 1):
            queries.append((i, i + 1))
    log(f"Query mode: {query_type}, yielded: {len(queries)} queries")
    descriptors, matches, frames = match_images(targets, circles, matching_query=queries, DESC_SIZE=desc_size)
    
    safe_queries = []

    for Q in tqdm(range(len(queries))):
        q = queries[Q]
        P1 = targets[q[0]]
        P2 = targets[q[1]]
        M = matches[Q][0]
        if len(M) == 0 or len(M.shape) == 1:
            logger.warning(f"Insufficient or mishapen matches in query {Q}: {M.shape}. Skipping")
            continue

        mask = matches[Q][1]

        DRAWING_C1 = descriptors[q[0]][-1][mask.astype(bool)]
        DRAWING_C2 = descriptors[q[1]][-1]
        
        C1 = descriptors[q[0]][0][mask.astype(bool)]
        C2 = descriptors[q[1]][0]
        
        t, b, s, count, mask = ransac_fn(C1, C2[M[:, 0].astype(int)], error=30)

        if np.sum(mask) >= min_correspondances:
            safe_queries.append((
                q,
                t
            ))
            if not match_dir is None:
                if type(match_dir) == str:
                    match_dir = Path(match_dir)

                c1 = draw_matches(P1, P2, M, DRAWING_C1.astype(int), DRAWING_C2.astype(int))
                c2 = draw_matches(P1, P2, M[mask], DRAWING_C1[mask].astype(int), DRAWING_C2.astype(int))

                cv2.imwrite(str(match_dir / f"match_{Q}.jpg"), c1[..., ::-1])
                cv2.imwrite(str(match_dir / f"ransac_match_{Q}.jpg"), c2[..., ::-1])
        else:
            log(f"Match ({q[0]}, {q[1]}) had insufficient matches after ransac. Before: {len(matches)}, After: {np.sum(mask)}")
    log(f"Saved all matching and RANSAC results. Found {len(safe_queries)} good matches.")
    return safe_queries, frames
            
def ransac_transform(A, B):
    assert(A.shape[0] >= 2)
    def solve(v1, v2, z1, z2):
        # AM = B
        # M = A^-1 B
        
        a = np.vstack([
            v1, v2
        ])

        b = np.vstack([
            z1, z2
        ])
        a_inv = np.linalg.inv(a)
        return a_inv @ b

    best_M = None
    best_score = np.inf
    log(f"Finding transform for points of shape: {A.shape}")
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i == j:
                continue
            try:
                M = solve(
                    A[i],
                    A[j],
                    B[i],
                    B[j]
                )
                print(M)
                score = np.linalg.norm(A @ M - B, axis=-1)
                score = np.median(score)
                if score < best_score:
                    best_M = M
                    best_score = score
            except:
                logger.warning(f"Failed to find M for {i}, {j}")
    if best_M is None:
        logger.warning("Failed to find good ransac result...")
    return best_M,  best_score

from enum import Enum
class TransformMode(Enum):
    Linear = "Linear2D"
    Rectified = "Rectified"

def lstsq_ransac_2d_transform(offsets, translations, samples_per_test, tests, threshold=40):
    best_score = np.inf
    best_count = -np.inf
    best = None
    for _ in tqdm(range(tests)):
        idxs = np.random.randint(0, offsets.shape[0], samples_per_test)
        try:
            s = np.mean(translations[idxs] / offsets[idxs][::-1], axis=0)
            M = np.diag(s[::-1])[..., ::-1] # Reflect around y=x
            score = np.linalg.norm(offsets @ M - translations, axis=-1)
            count = np.mean(score < threshold)
            score = np.median(score)
            if count > best_count or (count == best_count and score < best_score):
                best_count = count
                best_score = score
                best = M
        except:
            logger.warning("Least squares failed.")
    return best, best_count, best_score


def lstsq_ransac_transform(offsets, translations, samples_per_test, tests, threshold=40):
    best_score = np.inf
    best_count = -np.inf
    best = None
    for _ in tqdm(range(tests)):
        idxs = np.random.randint(0, offsets.shape[0], samples_per_test)
        try:
            M = np.linalg.lstsq(offsets[idxs], translations[idxs])[0]
            score = np.linalg.norm(offsets @ M - translations, axis=-1)
            count = np.mean(score < threshold)
            score = np.median(score)
            if count > best_count or (count == best_count and score < best_score):
                best_count = count
                best_score = score
                best = M
        except:
            logger.warning("Least squares failed.")
    return best, best_count, best_score


def map_coords_to_image(targets, safe_transforms, arc_target_coords, mode: TransformMode = TransformMode.Rectified):
    offsets = np.array([arc_target_coords[q[1]] - arc_target_coords[q[0]] for q, _ in safe_transforms])
    translations = np.array([t for _, t in safe_transforms])
    if mode == TransformMode.Linear:
        M, c, s = lstsq_ransac_transform(offsets, translations, 6, 10_000)
    elif mode == TransformMode.Rectified:
        M, c, s = lstsq_ransac_2d_transform(offsets, translations, 6, 10_000)
    else:
        raise "Transform mode not configured"
    # M, s = ransac_transform(offsets, translations)
    log(f"Found transform {M}\nMode: {mode}\nMedian Error: {s}")

    # for idx, q in enumerate(tqdm(safe_transforms)):
    #     q = q[0]
    #     t = offsets[idx] @ M
    #     im1 = np.mean(np.asarray(Image.open(targets[q[0]])), axis=-1)
    #     warped = cv2.warpAffine(im1, np.array([[1, 0, t[0]], [0, 1, t[1]]]), im1.shape[:2][::-1])
    #     im2 = np.mean(np.asarray(Image.open(targets[q[1]])), axis=-1)
    #     result = np.dstack([
    #         warped,
    #         im2,
    #         np.zeros_like(im1)
    #     ]).astype(np.uint8)
    #     cv2.imwrite(f"rewarping_{idx}.jpg", result.astype(np.uint8))
    log(f"Saved target transforms")
    return M, offsets, translations