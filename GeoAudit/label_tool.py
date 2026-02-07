import cv2
import numpy as np
import os
import glob
import argparse

# --- CONFIGURATION ---
DRAW_THICKNESS = 4
COLOR_ROAD = (255, 255, 255) # White for road
COLOR_BG = (0, 0, 0) # Black for background

# Global variables for mouse callback
drawing = False
last_point = None
current_mask = None
current_image = None
display_image = None

def mouse_callback(event, x, y, flags, param):
    global drawing, last_point, current_mask, display_image, current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw on the mask
            cv2.line(current_mask, last_point, (x, y), 255, DRAW_THICKNESS)
            # Draw on the display image for visual feedback (red overlay)
            cv2.line(display_image, last_point, (x, y), (0, 0, 255), DRAW_THICKNESS)
            last_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def label_images(input_dir, output_img_dir, output_mask_dir):
    global current_mask, display_image, current_image

    # Create directories if not exist
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # Get all images
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(input_dir, ext)))
    
    images = sorted(images)
    
    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(images)} images.")
    print("Controls:")
    print("  Left Mouse: Draw Road")
    print("  's': Save and Next")
    print("  'c': Clear current mask")
    print("  'q': Quit")
    print("  'n': Skip/Next")

    cv2.namedWindow("Labeling Tool")
    cv2.setMouseCallback("Labeling Tool", mouse_callback)

    for i, img_path in enumerate(images):
        basename = os.path.basename(img_path)
        print(f"[{i+1}/{len(images)}] Processing: {basename}")
        
        # Check if already exists
        save_mask_path = os.path.join(output_mask_dir, os.path.splitext(basename)[0] + ".png")
        if os.path.exists(save_mask_path):
            print(f"  -> Already exists, skipping (delete from {output_mask_dir} to redo)")
            continue

        # Load image
        original_img = cv2.imread(img_path)
        if original_img is None:
            print("  -> Error reading image")
            continue
            
        # Resize for consistent training data? Optional.
        # For now, keep original size.
        
        current_image = original_img.copy()
        display_image = original_img.copy()
        
        # Initialize blank mask (black)
        h, w = original_img.shape[:2]
        current_mask = np.zeros((h, w), dtype=np.uint8)

        while True:
            # Blend mask with image for visualization
            # We already draw red lines on display_image in the callback for speed
            # But let's refresh to show the clean underlying image + mask overlay periodically if needed
            # For simplicity, we just show display_image which collects the red lines
            
            cv2.imshow("Labeling Tool", display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # Save
                # 1. Save Image to training folder (converted to png)
                save_img_path = os.path.join(output_img_dir, os.path.splitext(basename)[0] + ".png")
                cv2.imwrite(save_img_path, original_img)
                
                # 2. Save Mask
                cv2.imwrite(save_mask_path, current_mask)
                print(f"  -> Saved {save_img_path} and {save_mask_path}")
                break
                
            elif key == ord('c'):
                # Clear
                print("  -> Cleared")
                current_mask = np.zeros((h, w), dtype=np.uint8)
                display_image = original_img.copy()
                
            elif key == ord('n'):
                # Skip
                print("  -> Skipped")
                break
                
            elif key == ord('q'):
                print("Exiting...")
                return

    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Road Labeling Tool")
    parser.add_argument("--input", "-i", type=str, default="raw_data", help="Folder containing raw images")
    parser.add_argument("--output_images", "-oi", type=str, default="data/train/images", help="Folder to save training images")
    parser.add_argument("--output_masks", "-om", type=str, default="data/train/masks", help="Folder to save training masks")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        os.makedirs(args.input)
        print(f"Created input directory '{args.input}'. Please put your raw satellite images there and run this script again.")
    else:
        label_images(args.input, args.output_images, args.output_masks)
