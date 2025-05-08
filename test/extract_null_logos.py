import os
import json
import shutil
from pathlib import Path

# Configuration
FEATURES_PATH = "output/features/logo_features.json"  # Adjust if needed
LOGO_IMAGE_DIR = "output/logos"      # Where the original logos are
OUTPUT_DIR = "test/logos_w_null_features"     # Where to copy null-feature logos

def extract_null_feature_logos():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(FEATURES_PATH, "r") as f:
        data = json.load(f)

    count = 0
    for entry in data:
        if not entry.get("features"):  # catches null, None, or missing keys
            logo_path = Path(entry["path"])
            if logo_path.exists():
                target = Path(OUTPUT_DIR) / logo_path.name
                shutil.copy(logo_path, target)
                count += 1
            else:
                print(f"Missing file: {logo_path}")

    print(f"\nCopied {count} logos with null features to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    extract_null_feature_logos()