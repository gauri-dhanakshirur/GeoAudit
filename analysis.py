import numpy as np

def identify_material(original_img, mask):
    road_pixels = original_img[mask > 0]
    if len(road_pixels) == 0: return "Unknown"
    avg_color = np.mean(road_pixels, axis=0)
    
    # Classification for Bituminous, Concrete, and Earthen roads
    if avg_color[0] > 160 and avg_color[1] > 160:
        return "Concrete / White-topped"
    elif avg_color[0] > avg_color[2] + 15:
        return "Earthen Road"
    else:
        return "Bituminous / Black-top"

def calculate_road_length(skeleton, gsd=0.3):
    # gsd 0.3 represents 30 cm GSD from Cartosat-2/3
    pixel_count = np.sum(skeleton > 0)
    length_km = (pixel_count * gsd) / 1000
    return round(length_km, 3)