# step3_tiling.py

import cv2
import os

# -------- CONFIG --------
INPUT_IMAGE = "raw_images/bhuvan_preprocessed.png"
OUTPUT_DIR = "tiles"

TILE_SIZE = 256
OVERLAP = 32

# ------------------------

def main():
    if not os.path.exists(INPUT_IMAGE):
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img = cv2.imread(INPUT_IMAGE)
    h, w, _ = img.shape
    print("Image height:", h)
    print("Image width:", w)
    print("Tile size:", TILE_SIZE)

    stride = TILE_SIZE - OVERLAP
    tile_id = 0

    for y in range(0, h - TILE_SIZE + 1, stride):
        for x in range(0, w - TILE_SIZE + 1, stride):
            tile = img[y:y + TILE_SIZE, x:x + TILE_SIZE]

            tile_name = f"tile_{tile_id}_x{x}_y{y}.png"
            tile_path = os.path.join(OUTPUT_DIR, tile_name)

            cv2.imwrite(tile_path, tile)
            tile_id += 1

    print(f"Total tiles generated: {tile_id}")
    print(f"Tiles saved in folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
