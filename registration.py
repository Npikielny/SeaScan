import cv2, numpy as np, matplotlib.pyplot as plt, glob
from tqdm import tqdm
from loguru import logger
log = logger.debug

def create_descriptors(image, centers, descriptor_radius):
    img = image.astype(np.float32)
    
    starts = centers - descriptor_radius
    ends = centers + descriptor_radius

    mask = np.mean(np.bitwise_and(starts >= 0, ends < img.shape[:2][::-1]), axis=-1) >= 1
    log(f"Keeping {np.mean(mask) * 100:.2f}% of descriptors ({np.sum(mask)})")
        
    descriptors = np.array([
        img[
            s[1]:e[1],
            s[0]:e[0]
        ] for s, e in zip(starts[mask], ends[mask])
    ])
    means = np.mean(descriptors, axis=(1, 2))
    std = np.std(descriptors, axis=(1,2))
    descriptors -= means[:, np.newaxis, np.newaxis, :]
    descriptors /= std[:, np.newaxis, np.newaxis, :]
    return np.linalg.norm(descriptors, axis=-1), mask

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
    scores = np.array([find_best_score(d, desc2s) for d in desc1s])
    while len(np.unique(scores[:, 0])) != len(desc2s) and len(np.unique(scores[:, 0])) != len(desc1s):
        # print("REMOVING DOUBLES", len(np.unique(scores[:, 0])), len(desc2s), len(desc1s))
        remaining = np.arange(desc2s.shape[0])
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
def match_images(paths, circlesclea, matching_query=None, DESC_SIZE=60):
    descriptors = []

    for path, res in tqdm(zip(paths, circles), "Creating descriptors"):
        image = np.asarray(Image.open(path))
        centers = (np.round(res[:, :2])).astype(int)
        desc, mask = create_descriptors(image, centers, DESC_SIZE)
        descriptors.append(
            (
                centers[mask],
                desc, 
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
    return descriptors, results
        
def draw_matches(path1, path2, matches, centers1, centers2):
    im1 = np.asarray(Image.open(path1))
    im2 = np.asarray(Image.open(path2))
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = im1
    canvas[:h2, w1:w1 + w2] = im2
    
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
    # plt.imshow(canvas)
    # _ = plt.axis('off')
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

def get_image_transform(gimball_pitch, image_shape):
    ctr = np.array(image_shape[:2][::-1]) / 2
    gimball_pitch = gimball_pitch / 180 * np.pi 
    c = np.cos(gimball_pitch)
    s = np.sin(gimball_pitch)
    rot = np.array([
        [c, -s],
        [s, c]
    ])
    return ctr, rot # Convert to inverse

def image_to_world(coord, ctr, rot):
    return (coord - ctr) @ rot # Right multiply is inverse

def world_to_image(coord, ctr, rot):
    return coord @ rot.T + ctr

def get_good_transforms(targets, circles, ransac_fn=ransac_d, min_correspondances=3, desc_size=100):
    queries = []
    for i in range(len(targets) - 1):
        queries.append((i, i + 1))
    descriptors, matches = match_images(targets, circles, matching_query=queries, DESC_SIZE=desc_size)
    # t, b, s, count, mask = ransac_d(C1, C2[M[:, 0].astype(int)], error=30)
    safe_queries = []

    for Q in tqdm(range(len(queries))):
        q = queries[Q]
        P1 = targets[q[0]]
        P2 = targets[q[1]]
        M = matches[Q][0]
        mask = matches[Q][1]
        
        C1 = descriptors[q[0]][0][mask.astype(bool)]
        C2 = descriptors[q[1]][0]
        
        c = draw_matches(P1, P2, M, C1, C2)
        cv2.imwrite(f"match_{Q}.jpg", c[..., ::-1])
        t, b, s, count, mask = ransac_d(C1, C2[M[:, 0].astype(int)], error=30)
        # t, b, s, count, mask = ransac_e(C1, C2[M[:, 0].astype(int)])
    
        c = draw_matches(P1, P2, M[mask], C1[mask], C2)
        cv2.imwrite(f"ransac_match_{Q}.jpg", c[..., ::-1])

        if np.sum(mask) >= min_correspondances:
            safe_queries.append((
                q,
                t
            ))
    log("Saved all matching and RANSAC results.")
    return safe_queries
            
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
            score = np.linalg.norm(A @ M - B, axis=-1)
            score = np.median(score)
            if score < best_score:
                best_M = M
                best_score = score
        except:
            print(f"Failed to find M for {i}, {j}")
    return best_M,  best_score

def map_coords_to_image(targets, safe_transforms, arc_target_coords):
    offsets = np.array([arc_target_coords[q[1]] - arc_target_coords[q[0]] for q, _ in safe_transforms])
    translations = np.array([t for _, t in safe_transforms])
    M, s = ransac_transform(offsets, translations)
    log(f"Found transform {M} with error: {s}")

    for idx, q in enumerate(tqdm(safe_transforms)):
        q = q[0]
        t = offsets[idx] @ M
        im1 = np.mean(np.asarray(Image.open(targets[q[0]])), axis=-1)
        warped = cv2.warpAffine(im1, np.array([[1, 0, t[0]], [0, 1, t[1]]]), im1.shape[:2][::-1])
        im2 = np.mean(np.asarray(Image.open(targets[q[1]])), axis=-1)
        result = np.dstack([
            warped,
            im2,
            np.zeros_like(im1)
        ]).astype(np.uint8)
        cv2.imwrite(f"rewarping_{idx}.jpg", result.astype(np.uint8))
    log(f"Saved target transforms")
    return M