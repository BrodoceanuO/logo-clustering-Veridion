# config.py
import os

FEATURE_FLAGS = {
    "edge_density": True,
    "dominant_color": True,
    "hue_family": False,
    "aspect_ratio": True,
    "sharpness": False, # no results, can't cluster with any threshold
    "shape": False # no results, can't cluster with any threshold
}

# input and output directories
#INPUT_DIR = "input"
#OUTPUT_DIR = "output"
INPUT_DIR = "input"
OUTPUT_DIR = "output"
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

DISTANCE_THRESHOLD = 0.1

DEFAULT_DISTANCE = 1

# Optional feature weights (relative importance)
FEATURE_WEIGHTS = {
    "edge_density": 0.2,
    "dominant_color": 0.2,
    "hue_family": 1.0, # set to off
    "aspect_ratio": 0.5,
    "sharpness": 1.0, # set to off
    "shape": 1.0 # set to off
}

# Paths for feature extraction
LOGO_PARQUET_PATH = os.path.join(INPUT_DIR, "logos.snappy.parquet")
LOGO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "logos")

# Paths for propagation and logging
LOGO_LOG_RESULTS_PATH = os.path.join(LOGS_DIR, "logo_results.json")
LOGO_DIR = LOGO_OUTPUT_PATH

# Paths for clustering
NULL_LOG_PATH = os.path.join(LOGS_DIR, "null_feature_logos.json")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "features", "logo_features.json")
CLUSTERS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "clusters", "clusters_filtered.json")