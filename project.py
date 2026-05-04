import numpy as np

from loguru import logger
log = logger.debug

from pathlib import Path

class Project(object):
    def __init__(self, path):
        self.path = path
        if Path(path).exists():
            self.data = np.load(path, allow_pickle=True).item()
        else:
            self.data = {}

    def initialize_dataset(self, images, targets, coords):
        self["images"] = images
        self["targets"] = targets
        self["coords"] = coords
        self.write()

    def stash(self, value_dict):
        for k in value_dict:
            self[k] = value_dict[k]
        self.write()

    def save_features(self, feature_data):
        self["features"] = feature_data
        self.write()

    def save_matches(self, match_data):
        self["matches"] = match_data
        self.write()

    def save_triangulation(self, transform, offsets, translations):
        self["transform"] = transform
        self["offsets"] = offsets 
        self["translations"] = translations
        self.write()

    def save_vignette_mask(self, mask):
        self["v_mask"] = mask
        self.write()

    def __getitem__(self, key):
        return self.data.get(key)

    def __setitem__(self, instance, value):
        self.data[instance] = value
        
    def write(self):
        np.save(self.path, self.data, allow_pickle=True)
        log(f"Saved file to {self.path}")

    def keys(self):
        return self.data.keys()

