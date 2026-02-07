import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from advanced_model import AdvancedRoadModel

import argparse
import sys

# Configuration
MODEL_PATH = "advanced_road_model.pth"
# TEST_IMAGE_DIR = "data/train/images" # Removed hardcoded dir
OUTPUT_FILENAME = "test_result_skeleton2.png"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256

def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img (a Mat object)"""
    img = img.copy()
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    while True:
        open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open_img)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

def calculate_length_km(skeleton, gsd_meters=0.3):
    """ Calculate total length of skeleton in km """
    # Count pixels
    pixel_count = cv2.countNonZero(skeleton)
    # Total meters
    length_meters = pixel_count * gsd_meters
    return length_meters / 1000.0

def identify_material(img_rgb, mask):
    """
    Identify road material based on Color (HSV) and Brightness.
    Supported types:
    - Bituminous (Asphalt): Dark Gray/Black (Low Saturation, Low Brightness)
    - Concrete: Light Gray/White (Low Saturation, High Brightness)
    - Mud/Dirt: Brown/Orange (Higher Saturation, Orange Hue)
    """
    # Extract only road pixels
    road_pixels_rgb = img_rgb[mask == 255]
    
    if len(road_pixels_rgb) == 0:
        return "Unknown"
        
    # Convert pixels to HSV
    # We need to reshape to (N, 1, 3) for cv2.cvtColor to work on a list of pixels
    road_pixels_rgb_reshaped = road_pixels_rgb.reshape(-1, 1, 3)
    road_pixels_hsv = cv2.cvtColor(road_pixels_rgb_reshaped, cv2.COLOR_RGB2HSV)
    road_pixels_hsv = road_pixels_hsv.reshape(-1, 3) # Back to (N, 3)
    
    # Calculate average HSV
    avg_h = np.mean(road_pixels_hsv[:, 0]) # Hue (0-179)
    avg_s = np.mean(road_pixels_hsv[:, 1]) # Saturation (0-255)
    avg_v = np.mean(road_pixels_hsv[:, 2]) # Value/Brightness (0-255)
    
    print(f"DEBUG: Avg HSV: H={avg_h:.1f}, S={avg_s:.1f}, V={avg_v:.1f}")
    
    # Logic:
    # 1. Mud/Dirt is colorful (Brown/Orange/Reddish) -> Significant Saturation
    # Threshold for "Color" vs "Grayscale": Saturation > 25 (approx 10%)
    if avg_s > 30: 
        # Check Hue for Brown/Orange (Red is 0-10 or 170-180, Orange/Yellow is 10-30)
        # Soil is typically roughly Orange-ish.
        if (avg_h >= 0 and avg_h < 40) or (avg_h > 160):
            return "Mud/Dirt"
        else:
            return "Unknown (Colored)" 
            
    # 2. If Low Saturation, it's Gray (Asphalt or Concrete)
    else:
        # Check Brightness
        if avg_v < 110:
            return "Bituminous (Asphalt)"
        else:
            return "Concrete"

def test():
    parser = argparse.ArgumentParser(description="Test Advanced Road Model on an image")
    parser.add_argument("image_path", nargs="?", help="Path to the input image")
    args = parser.parse_args()

    # Determine image path
    img_path = args.image_path
    
    if not img_path:
        # Fallback to default if no argument provided
        test_dir = "data/train/images"
        images = sorted(glob.glob(os.path.join(test_dir, "*.png")))
        if images:
            img_path = images[0]
            print(f"No image provided. Using default sample: {img_path}")
        else:
            print("Error: No image provided and no samples found in data/train/images")
            return

    print(f"Using device: {DEVICE}")

    # 1. Load Model
    model = AdvancedRoadModel(num_classes=1).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"Loaded model weights from {MODEL_PATH}")
        except Exception as e:
             print(f"Error loading model: {e}")
             return
    else:
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    model.eval()

    print(f"Testing on image: {img_path}")

    # 3. Preprocess
    original_image = cv2.imread(img_path)
    if original_image is None:
        print(f"Could not read image {img_path}. Check the path.")
        return
        
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    input_image = cv2.resize(original_image, (IMG_SIZE, IMG_SIZE))
    
    # Normalize and convert to tensor
    tensor_img = input_image.astype(np.float32) / 255.0
    tensor_img = torch.from_numpy(tensor_img).permute(2, 0, 1).unsqueeze(0) # 1, C, H, W
    tensor_img = tensor_img.to(DEVICE)

    # 4. Inference
    with torch.no_grad():
        output = model(tensor_img)
        pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()

    # 5. Post-process
    # Threshold
    binary_mask = (pred_prob > 0.5).astype(np.uint8) * 255
    
    # Skeletonize
    skel = skeletonize(binary_mask)
    
    # Analysis
    road_type = identify_material(input_image, binary_mask)
    length_km = calculate_length_km(skel)
    
    print(f"Detected Road Type: {road_type}")
    print(f"Estimated Length: {length_km:.4f} km")

    # 6. Visualize
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(input_image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_prob, cmap="gray")
    plt.title("Predicted Probability")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    # Overlay skeleton on original
    combined = input_image.copy()
    # Make skeleton cyan
    combined[skel == 255] = [0, 255, 255]
    
    plt.imshow(combined)
    plt.title(f"Type: {road_type} | Len: {length_km:.3f} km")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME)
    print(f"Result saved to {OUTPUT_FILENAME}")
    # plt.show() # Uncomment if you want pop-up

if __name__ == "__main__":
    test()
