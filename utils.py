import json
import os
from collections import Counter
import cv2
import numpy as np
from PIL import Image

def print_stats(results, title="", output_file=None):
    total = len(results)
    direct_success = sum(1 for r in results.values() if r["status"] == "success" and r["source"] != "propagated")
    propagated_success = sum(1 for r in results.values() if r["status"] == "success" and r["source"] == "propagated")
    total_success = direct_success + propagated_success
    failures = total - total_success

    # Count failure reasons
    failure_reasons = Counter()
    for r in results.values():
        if r["status"] != "success":
            reason = r.get("reason", "unknown")
            failure_reasons[reason] += 1

    summary = {
        "title": title,
        "total_domains": total,
        "direct_success": direct_success,
        "propagated_success": propagated_success,
        "total_success": total_success,
        "failures": failures,
        "percentages": {
            "direct": round(direct_success / total * 100, 2),
            "propagated": round(propagated_success / total * 100, 2),
            "success_total": round(total_success / total * 100, 2),
            "failures": round(failures / total * 100, 2),
        },
        "failure_reasons": dict(failure_reasons)
    }

    print(f"\n📊 {title}")
    print(f"📦 Total domains: {total}")
    print(f"🟢 Direct extractions:     {direct_success} ({summary['percentages']['direct']}%)")
    print(f"🟡 Propagated extractions: {propagated_success} ({summary['percentages']['propagated']}%)")
    print(f"✅ Total success:          {total_success} ({summary['percentages']['success_total']}%)")
    print(f"🔴 Total failures:         {failures} ({summary['percentages']['failures']}%)")

    if failures:
        print("\n🧯 Failure reasons:")
        for reason, count in failure_reasons.items():
            print(f"  - {reason}: {count}")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n📝 Stats saved to {output_file}")

def get_edge_background_mask(img):
    """
    Returns a binary mask that separates foreground from background
    based on majority edge color, even for images with alpha.
    Mask with white (255) = foreground, black (0) = background
    """
    try:
        # If alpha channel is present, discard it
        if img.shape[-1] == 4:
            print("Stripping alpha channel for background color masking")
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[:2]

        # Gather edge pixels (top, bottom, left, right)
        edge_pixels = np.vstack([
            img[0, :],         # top edge
            img[-1, :],        # bottom edge
            img[:, 0],         # left edge
            img[:, -1]         # right edge
        ])

        # Estimate background color
        bg_color = np.median(edge_pixels, axis=0)
        print(f"Estimated background color: {bg_color}")
        bg_color = np.round(bg_color).astype(np.uint8)

        # Tile to full image
        bg_tile = np.tile(bg_color[np.newaxis, np.newaxis, :], (h, w, 1))

        # Color distance
        diff = cv2.absdiff(img, bg_tile)
        dist = np.linalg.norm(diff, axis=2)

        # Threshold to get foreground mask
        _, mask = cv2.threshold(dist.astype(np.uint8), 60, 255, cv2.THRESH_BINARY)

        # _, mask = cv2.threshold(dist, 60, 255, cv2.THRESH_BINARY)
        # mask = mask.astype(np.uint8)

        cv2.imwrite("debug_outputs/alpha_mask_util.png", mask)

        # Optional cleanup
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    except Exception as e:
        print(f"⚠️ Error generating background mask: {e}")
        return np.ones(img.shape[:2], dtype=np.uint8) * 255  # fallback: all foreground
    
def get_alpha_mask(img):
    """
    Returns a mask based on the alpha channel transparency
    """
    if img is None or len(img.shape) != 3 or img.shape[2] != 4:
        return None  # No alpha channel

    alpha = img[:, :, 3]
    _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    return mask

def get_foreground_mask(img):
    """
    Chooses the right foreground masking function.
    Returns: (mask, strategy)
    strategy = 'alpha' | 'background'
    """
    if img.shape[-1] == 4:
        alpha = img[:, :, 3]
        if np.any(alpha < 255):
            print("alpha")
            return get_alpha_mask(img), "alpha"
        else:
            print("background (alpha present but fully opaque)")
    else:
        print("background (no alpha)")

    return get_edge_background_mask(img), "background"

def load_image(path):
    """
    Loads an image and ensures it's in a usable RGB(A) format.
    Handles grayscale, 1-channel, and broken cases.
    Returns a NumPy array with 3 or 4 channels.
    """
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if img is not None and img.size != 0:
            # Handle grayscale or 1-channel OpenCV load
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Allow BGR (3-channel) or BGRA (4-channel) to pass through
            return img

        # Fallback: use PIL and ensure 4 channels
        pil_img = Image.open(path).convert("RGBA")
        img = np.array(pil_img)

        # Ensure it's (H, W, 4)
        if img.shape[2] != 4:
            raise ValueError("PIL fallback failed to produce 4-channel image.")

        return img

    except Exception as e:
        print(f"⚠️ Failed to load image {path}: {e}")
        return None