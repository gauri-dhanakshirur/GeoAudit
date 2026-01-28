import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from advanced_model import AdvancedRoadModel
from image_processor import occlusion_aware_clean, get_skeleton
from analysis import identify_material, calculate_road_length

# --- CONFIGURATION ---
IMG_PATH = "bs9.png" # Using same image as main.py
MODEL_PATH = "advanced_road_model.pth"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 256

def load_advanced_model():
    print(f"Loading Advanced Model on {DEVICE}...")
    model = AdvancedRoadModel(num_classes=1)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Loaded trained weights.")
    except Exception as e:
        print(f"Warning: Could not load weights ({e}). Using random initialization for demo.")
        print("NOTE: Results will be noise until trained!")
    
    model.to(DEVICE).eval()
    return model

def predict_full_image_advanced(model, img_rgb, patch_size):
    h, w, _ = img_rgb.shape
    
    # Pad to multiple of patch size
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    img_padded = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    
    new_h, new_w, _ = img_padded.shape
    full_mask = np.zeros((new_h, new_w))
    
    print("Running Inference with D-LinkNet + Attention + Strip Convs...")
    with torch.no_grad():
        for y in range(0, new_h, patch_size):
            for x in range(0, new_w, patch_size):
                patch = img_padded[y:y+patch_size, x:x+patch_size]
                
                # Preprocess
                tensor = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).float() / 255.0
                tensor = tensor.to(DEVICE)
                
                # Inference
                output = model(tensor)
                prediction = torch.sigmoid(output).squeeze().cpu().numpy()
                
                full_mask[y:y+patch_size, x:x+patch_size] = prediction

    # Crop back
    final_mask = full_mask[:h, :w]
    return final_mask

def main():
    # 1. Load Image
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"Error: {IMG_PATH} not found.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Prediction
    model = load_advanced_model()
    raw_prob = predict_full_image_advanced(model, img_rgb, PATCH_SIZE)
    
    # Thresholding (Adjust based on training convergence, 0.5 is standard)
    binary_mask = (raw_prob > 0.5).astype(np.uint8) * 255

    # 3. Processing (Occlusion Smoothing)
    cleaned = occlusion_aware_clean(binary_mask)
    skeleton = get_skeleton(cleaned)

    # 4. Analysis
    material = identify_material(img_rgb, cleaned)
    # Estimate GSD (0.3m standard for high res sats)
    km_length = calculate_road_length(skeleton, 0.3)

    # 5. Visualization
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Input")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Advanced Model Prediction\n(Attention + D-LinkNet)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_rgb)
    plt.contour(skeleton, colors='lime', linewidths=0.5)
    plt.title(f"Network & Analysis\nType: {material} | Len: {km_length}km")
    plt.axis('off')
    
    output_filename = "advanced_output_demo.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    print(f"Result saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()
