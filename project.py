import numpy as np

from loguru import logger
log = logger.debug
def write_project(
    to_file: str, 
    target: str, 
    images: [str],
    mapping: np.ndarray, 
    v_mask: np.ndarray 
):
    obj = {
        "M": mapping,
        "target": target,
        "images": images,
        "v_mask": v_mask
    }
    np.save(to_file, obj, allow_pickle=True)
    log(f"Saved file to {to_file}")

def read_project(
    from_file: str
):
    obj = np.load(from_file, allow_pickle=True)
    return obj["M"], obj["target"], obj["images"], obj["v_mask"]