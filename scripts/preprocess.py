# step2_preprocess.py

import cv2
import os

# -------- CONFIG --------
INPUT_IMAGE = "raw_images/bhuvan_full.png"
OUTPUT_IMAGE = "raw_images/bhuvan_preprocessed.png"
APPLY_CLAHE = True
# ------------------------

def apply_clahe(image):
    """
    Apply CLAHE on the L channel to improve contrast
    without changing colors.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return enhanced


def main():
    if not os.path.exists(INPUT_IMAGE):
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

    img = cv2.imread(INPUT_IMAGE)

    if img is None:
        raise ValueError("Failed to load image. Check file format.")

    print(f"Loaded image shape: {img.shape}")

    if APPLY_CLAHE:
        img = apply_clahe(img)
        print("CLAHE contrast enhancement applied")

    cv2.imwrite(OUTPUT_IMAGE, img)
    print(f"Preprocessed image saved to: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
