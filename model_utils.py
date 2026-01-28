import torch
import numpy as np
import segmentation_models_pytorch as smp

def get_model(device="cpu"):
    # Using ResNet-34 as the feature extractor for road textures
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    model.to(device).eval()
    return model

def predict_full_image(model, img_padded, patch_size, device="cpu"):
    h, w, _ = img_padded.shape
    full_mask = np.zeros((h, w))
    with torch.no_grad():
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = img_padded[y:y+patch_size, x:x+patch_size]
                tensor = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).float() / 255.0
                output = torch.sigmoid(model(tensor.to(device)))
                full_mask[y:y+patch_size, x:x+patch_size] = output.squeeze().cpu().numpy()
    return full_mask