import os
import json
import math
from collections import defaultdict
import config
from tqdm import tqdm
    
def load_features_from_json():
    """Load logo features from a JSON file, filter enabled features, and skip invalid entries."""
    with open(config.FEATURES_PATH, "r") as f:
        raw_entries = json.load(f)

    filtered_entries = []
    null_entries = []

    for entry in raw_entries:
        domain = entry.get("domain") or entry.get("filename")
        favicon_fallback = entry.get("favicon_fallback", False)
        all_features = entry.get("features")

        if not all_features:
            null_entries.append({"domain": domain, "reason": "Missing or null features"})
            continue

        selected_features = {
            key: all_features[key]
            for key, enabled in config.FEATURE_FLAGS.items()
            if enabled and key in all_features
        }

        filtered_entries.append({
            "domain": domain,
            "favicon_fallback": favicon_fallback,
            "features": selected_features
        })

    # Write skipped logos to log file
    null_log_path = os.path.join(config.LOGS_DIR, "null_feature_logos.json")
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    with open(null_log_path, "w") as f:
        json.dump(null_entries, f, indent=2)

    print(f"Skipped {len(null_entries)} logos with missing features (saved to {null_log_path})")
    return filtered_entries

def try_parse(value):
    """Try converting a string value to a float. Leave it unchanged if that fails."""
    try:
        return float(value)
    except ValueError:
        return value

def distance_edge_density(val1, val2):
    if val1 is None or val2 is None:
        return 1.0
    return abs(val1 - val2)

def distance_aspect_ratio(val1, val2):
    if val1 is None or val2 is None:
        return 1.0
    return abs(val1 - val2)

def distance_dominant_color(rgb1, rgb2):
    if not (isinstance(rgb1, list) and isinstance(rgb2, list) and len(rgb1) == 3 and len(rgb2) == 3):
        return 1.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2))) / 441.67  # normalized RGB dist

def distance_hue_family(hue1, hue2):
    return 0.0 if hue1 == hue2 else 1.0

FEATURE_DISTANCE_FUNCTIONS = {
"edge_density": distance_edge_density,
"aspect_ratio": distance_aspect_ratio,
"dominant_color": distance_dominant_color,
"hue_family": distance_hue_family,
}

def compute_feature_distance(value1, value2, feature_name):
    """Dispatch to the correct feature-specific distance function."""

    distance_func = FEATURE_DISTANCE_FUNCTIONS.get(feature_name)

    if distance_func:
        return distance_func(value1, value2)
    else:
        # Fallback for unhandled features
        return 1.0

def compute_feature_distance_score(features_a, features_b):
    """Aggregate weighted feature distances between two logos."""
    total_weighted_distance = 0
    total_weight = 0

    for feature_name in features_a:
        if feature_name not in features_b:
            continue

        value_a = features_a[feature_name]
        value_b = features_b[feature_name]

        distance = compute_feature_distance(value_a, value_b, feature_name)
        weight = config.FEATURE_WEIGHTS.get(feature_name, 1.0)

        total_weighted_distance += distance * weight
        total_weight += weight

    if total_weight == 0:
        return 1.0  # no comparable features, treat as fully dissimilar

    return total_weighted_distance / total_weight

def compute_cluster_centroid(features_list):
    """
    Compute the centroid (average feature set) from a list of feature dictionaries.
    Only numeric and vector features are considered.
    Categorical features (e.g., strings) are ignored.
    """
    if not features_list:
        return {}

    centroid = {}
    feature_keys = features_list[0].keys()

    for key in feature_keys:
        values = [f[key] for f in features_list if key in f and f[key] is not None]

        if not values:
            continue

        first_value = values[0]

        if isinstance(first_value, (int, float)):
            centroid[key] = sum(values) / len(values)

        elif isinstance(first_value, list) and all(isinstance(x, (int, float)) for x in first_value):
            # Element-wise average for vectors like RGB or Hu moments
            zipped = zip(*values)
            centroid[key] = [sum(group) / len(group) for group in zipped]

        else:
            # Skip categorical or unsupported types
            continue

    return centroid

def cluster_logos(logo_entries):
    """Assign each logo to the most similar existing cluster based on cached centroids"""
    clusters = []
    cluster_features = []
    centroids = []

    print("Clustering logos...")
    for idx, logo in enumerate(tqdm(logo_entries, desc="Clustering", unit="logo")):
        logo_features = logo["features"]
        best_cluster_index = None
        best_distance = float("inf")

        # Compare to existing cached centroids
        for cluster_idx, centroid in enumerate(centroids):
            distance = compute_feature_distance_score(centroid, logo_features)

            if distance < config.DISTANCE_THRESHOLD and distance < best_distance:
                best_distance = distance
                best_cluster_index = cluster_idx

        if best_cluster_index is not None:
            # Assign to best cluster and update its feature set
            clusters[best_cluster_index].append(idx)
            cluster_features[best_cluster_index].append(logo_features)

            # Recompute that cluster's centroid only
            centroids[best_cluster_index] = compute_cluster_centroid(cluster_features[best_cluster_index])
        else:
            # Start a new cluster
            clusters.append([idx])
            cluster_features.append([logo_features])
            centroids.append(logo_features)  # Initial centroid is the logo itself

    return clusters

def save_cluster_results(clusters, logo_entries):
    """Save the clustered domain groups to a JSON file."""
    cluster_dict = {}
    os.makedirs(os.path.dirname(config.CLUSTERS_OUTPUT_PATH), exist_ok=True)

    for index, group_indices in enumerate(clusters, start=1):
        group_domains = [logo_entries[i]["domain"] for i in group_indices]
        cluster_dict[f"group_{index}"] = group_domains

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(config.CLUSTERS_OUTPUT_PATH, "w") as output_file:
        json.dump(cluster_dict, output_file, indent=2)

    print(f"Saved {len(cluster_dict)} clusters to {config.CLUSTERS_OUTPUT_PATH}")

def main():
    logo_entries = load_features_from_json()
    clusters = cluster_logos(logo_entries)
    save_cluster_results(clusters, logo_entries)

if __name__ == "__main__":
    main()