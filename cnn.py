import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# --- CONFIGURATION ---
INPUT_IMAGE = "bs.png"  # Your screenshot filename
PATCH_SIZE = 256
DEVICE = "cpu"  # Safe for demo; use "mps" for Mac GPU if available

# 1. LOAD MODEL (Pre-trained on ImageNet)
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
model.to(DEVICE).eval()

def process_demo():
    # 2. LOAD & PREPARE IMAGE
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print("Error: Image not found!")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    # Pad image to be divisible by PATCH_SIZE
    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
    img_padded = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    
    # 3. SLIDING WINDOW PREDICTION
    new_h, new_w, _ = img_padded.shape
    full_mask = np.zeros((new_h, new_w))

    with torch.no_grad():
        for y in range(0, new_h, PATCH_SIZE):
            for x in range(0, new_w, PATCH_SIZE):
                patch = img_padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                
                # Preprocess: Normalize and convert to Tensor
                tensor = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).float() / 255.0
                output = torch.sigmoid(model(tensor.to(DEVICE)))
                full_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = output.squeeze().cpu().numpy()

    # Crop back to original size
    final_raw_mask = full_mask[:h, :w]
    binary_mask = (final_raw_mask > 0.2).astype(np.uint8) * 255

    # --- UPDATED SECTION 4: OCCLUSION-AWARE CLEANING ---
    # --- UPDATED FOR ZOOMED-IN HIGH-RES IMAGE ---
    print("Applying Refined Processing for High-Res Zoom...")

    # 1. Standard Threshold: Zoomed-in pixels are usually more confident
    binary_mask = (final_raw_mask > 0.35).astype(np.uint8) * 255

    # 2. Light Dilation: Just enough to fill small texture gaps in the road surface
    # We reduce kernel from 7x7 to 3x3 to maintain 'geometrical regularity'
    kernel_dil = np.ones((3, 3), np.uint8)
    thick_mask = cv2.dilate(binary_mask, kernel_dil, iterations=1)

    # 3. Precise Morphological Closing:
    # We reduce kernel from 25x25 to 7x7. 
    # This still bridges tree occlusions but won't merge parallel roads together.
    kernel_bridge = np.ones((7, 7), np.uint8)
    closed_mask = cv2.morphologyEx(thick_mask, cv2.MORPH_CLOSE, kernel_bridge)

    # 4. Final Opening: Clean up the edges for a smooth skeleton
    kernel_smooth = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel_smooth)


    # 5. SKELETONIZATION (Vector Prep)
    # This addresses "Topological Correctness" by thinning the road network
    print("Generating Road Network Skeleton...")
    skeleton = skeletonize(cleaned > 0).astype(np.uint8) * 255

    # --- NEW: ROAD MATERIAL IDENTIFICATION LOGIC ---
    def identify_material(original_img, mask):
        # We only sample the color of pixels identified as 'road'
        road_pixels = original_img[mask > 0]
        
        if len(road_pixels) == 0:
            return "No Roads Detected"
        
        # Calculate average RGB values
        avg_color = np.mean(road_pixels, axis=0) 
        
        # Classification thresholds tailored for Indian road types
        # Concrete: High reflectance (bright white/grey)
        if avg_color[0] > 160 and avg_color[1] > 160:
            return "Concrete / White-topped"
        # Earthen: High red-channel dominance (brown/muddy)
        elif avg_color[0] > avg_color[2] + 15:
            return "Earthen Road"
        # Bituminous: Low reflectance (dark black/grey)
        else:
            return "Bituminous / Black-top"

    # Run identification on the padded image (cropped to original h, w)
    detected_material = identify_material(img_padded[:h, :w], cleaned)
    print(f"Analysis Result: {detected_material}")

    # 6. VISUALIZATION
    plt.figure(figsize=(20, 10))
    
    # Add the material classification to the main title for the demo
    plt.suptitle(f"Infrastructure Analysis | Material: {detected_material}", fontsize=22, y=0.98, color='darkblue')
    
    titles = [
        '1. Original Satellite View', 
        '2. Raw CNN Segments', 
        '3. Occlusion-Aware Cleaned', 
        '4. Final Network (Skeleton)'
    ]
    images = [img_rgb, binary_mask, cleaned, skeleton]
    
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], cmap='gray' if i > 0 else None)
        plt.title(titles[i], fontsize=14)
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("bangalore_demo_output6.png")
    print("Demo output saved as 'bangalore_demo_output6.png'")
    plt.show()

if __name__ == "__main__":
    process_demo()