import cv2
import matplotlib.pyplot as plt
from model_utils import get_model, predict_full_image
from image_processor import occlusion_aware_clean, get_skeleton
from analysis import identify_material, calculate_road_length

# Settings
IMG_PATH = "bs9.png"
GSD = 0.3 # Resolution for Indian Satellite Data

# 1. Prediction
model = get_model()
img = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
raw_mask = predict_full_image(model, img_rgb, 256)

# 2. Processing (Occlusion Smoothing)
cleaned = occlusion_aware_clean(raw_mask)
skeleton = get_skeleton(cleaned)

# 3. Analysis
material = identify_material(img_rgb, cleaned)
km_length = calculate_road_length(skeleton, GSD)

# 4. Visualization
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
# Overlay the skeleton in a glowing 'Neon Green'
plt.contour(skeleton, colors='lime', linewidths=0.5) 
plt.title(f"MoRTH Infrastructure Audit\nMaterial: {material} | Total Length: {km_length} KM", 
          fontsize=15, fontweight='bold', color='darkblue')
plt.axis('off')
plt.savefig("final_output7.png", dpi=300)
plt.show()

print(f"Demo Successful. Material: {material}, Length: {km_length}km")