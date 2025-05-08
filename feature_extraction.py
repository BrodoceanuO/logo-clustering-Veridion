import os
import json
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from pathlib import Path
from utils import get_foreground_mask, load_image
import config

def extract_edge_density(img):
    edges = cv2.Canny(img, 100, 200)
    edge_pixels = np.sum(edges > 0)
    total_pixels = img.shape[0] * img.shape[1]
    return edge_pixels / total_pixels

def extract_dominant_color(img, mask=None, k=3):
    try:
        if mask is not None:
            data = img[mask > 0]
        else:
            data = img.reshape((-1, 3))

        if data.size == 0:
            print("No pixels to analyze. Returning fallback color.")
            return [0, 0, 0]

        data = np.float32(data)

        _, labels, centers = cv2.kmeans(
            data, k, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            10, cv2.KMEANS_RANDOM_CENTERS
        )

        dominant = centers[Counter(labels.flatten()).most_common(1)[0][0]]

        return dominant.astype(int).tolist()

    except Exception as e:
        print(f"extract_dominant_color failed: {e}")
        return [0, 0, 0]

def get_hue_family(rgb):
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv

    if v < 40:
        return "black"
    elif s < 30:
        if v > 200:
            return "white"
        elif v > 100:
            return "gray"
        else:
            return "black"

    if h < 15 or h >= 165:
        return "red"
    elif h < 30:
        return "orange"
    elif h < 45:
        return "yellow"
    elif h < 85:
        return "green"
    elif h < 135:
        return "blue"
    else:
        return "purple"

def extract_aspect_ratio(mask):
    coords = cv2.findNonZero(mask)
    if coords is None:
        print("No foreground detected.")
        return None

    x, y, w, h = cv2.boundingRect(coords)
    if h == 0:
        print("Height is zero — invalid aspect ratio.")
        return None

    return w / h

def extract_sharpness_score(img, mask=None):
    cv2.imwrite("debug_outputs/foreground_extracted.png", img)
    print("Saved extracted foreground to debug_outputs/foreground_extracted.png")

    try:
        cv2.imwrite("debug_outputs/alpha_mask.png", mask)
        thresh = mask

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        total_angles = 0
        sharp_angles = 0

        debug_img = img[:, :, :3].copy() if img.shape[2] == 4 else img.copy()

        for contour in contours:
            if len(contour) < 3:
                continue

            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            cv2.polylines(debug_img, [approx], True, (0, 0, 255), 2)

            for i in range(1, len(approx) - 1):
                a = approx[i - 1][0]
                b = approx[i][0]
                c = approx[i + 1][0]

                ab = b - a
                cb = b - c

                norm_ab = np.linalg.norm(ab)
                norm_cb = np.linalg.norm(cb)

                if norm_ab == 0 or norm_cb == 0:
                    continue

                cos_angle = np.dot(ab, cb) / (norm_ab * norm_cb)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi

                # if total_angles < 10:
                #     print(f"Angle #{total_angles + 1}: {angle:.2f}°")

                if angle < 100:
                    sharp_angles += 1
                    cv2.circle(debug_img, tuple(b), 3, (255, 0, 0), -1)

                total_angles += 1

        if total_angles == 0:
            # print("No valid angles found.")
            return 0.5

        score = sharp_angles / total_angles
        # cv2.imwrite("debug_outputs/debug_overlay.png", debug_img)
        # print("Saved debug_overlay.png")
        # print(f"Total angles: {total_angles}, Sharp angles: {sharp_angles}")
        # print(f"Sharpness score: {score:.3f}")

        return round(score, 3)

    except Exception as e:
        print(f"Sharpness extraction error: {e}")
        return 0.0

def extract_shape_descriptor(img, mask=None):
    try:
        # cv2.imwrite("debug_outputs/foreground_extracted.png", img)
        # print("Saved extracted foreground to debug_outputs/foreground_extracted.png")

        # if mask is not None:
        #     img = cv2.bitwise_and(img, img, mask=mask)
        #     cv2.imwrite("debug_outputs/alpha_mask.png", mask)

        if len(img.shape) == 2 or img.shape[2] == 1:
            gray = img
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            debug_img = img.copy()

        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No contours found.")
            return [0] * 7

        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 1)
        cv2.drawContours(debug_img, [approx], -1, (0, 0, 255), 2)
        cv2.imwrite("debug_outputs/debug_overlay.png", debug_img)
        print("Saved debug_overlay.png")

        moments = cv2.moments(largest)
        hu = cv2.HuMoments(moments).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        return hu.round(4).tolist()

    except Exception as e:
        print(f"Hu moment extraction failed: {e}")
        return [0] * 7

def extract_features_for_logo(path):
    try:
        img = load_image(path)
        if img is None:
            raise ValueError("Could not load image")

        mask, strategy = get_foreground_mask(img)
        img_rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)

        edge_density = extract_edge_density(img)
        dominant_color = extract_dominant_color(img_rgb, mask)
        shape_descriptor = extract_shape_descriptor(img, mask)
        hue_family = get_hue_family(dominant_color)
        aspect_ratio = extract_aspect_ratio(mask)
        sharpness_score = extract_sharpness_score(img, mask)

        return {
            "edge_density": edge_density,
            "dominant_color": dominant_color,
            "hue_family": hue_family,
            "aspect_ratio": aspect_ratio,
            "sharpness": sharpness_score,
            "shape": shape_descriptor,
            "mask_strategy": strategy
        }

    except Exception as e:
        print(f"Failed to process {path.name}: {e}")
        return None

def extract_features_from_directory(input_dir = config.LOGO_DIR):
    """Extract features from all logo images in a directory."""
    input_path = Path(input_dir)
    print(input_path)
    supported_extensions = {".png", ".jpg", ".jpeg", ".ico", ".svg"}

    extracted_features = []

    for logo_path in sorted(input_path.glob("*")):
        if logo_path.suffix.lower() not in supported_extensions:
            continue

        print(f"Extracting features from {logo_path.name}...")

        try:
            features = extract_features_for_logo(logo_path)

            extracted_features.append({
                "filename": logo_path.name,
                "path": str(logo_path),
                "features": features
            })

            print(json.dumps(features, indent=2))  # Show extracted features for review

        except Exception as e:
            print(f"X Failed to process {logo_path.name}: {e}")

    return extracted_features

def save_features_to_json(features_list, output_path):
    """Save list of extracted features to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(features_list, f, indent=2)
    print(f"\nSaved features for {len(features_list)} logos to {output_path}")

def main():
    results = extract_features_from_directory(input_dir = config.LOGO_DIR)
    save_features_to_json(results, config.FEATURES_PATH)

if __name__ == "__main__":
    main()