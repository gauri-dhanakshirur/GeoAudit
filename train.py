import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import glob
import numpy as np
from advanced_model import AdvancedRoadModel
from loss import DiceBCELoss
import torch.optim as optim
from tqdm import tqdm

# --- CONFIGURATION ---
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
SAVE_PATH = "advanced_road_model.pth"

class RoadDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.masks = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize to fixed size for training
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # Normalize and convert to tensor
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        image = torch.from_numpy(image).permute(2, 0, 1) # C, H, W
        mask = torch.from_numpy(mask).unsqueeze(0)       # 1, H, W
        
        return image, mask

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Initialize Model
    model = AdvancedRoadModel(num_classes=1).to(DEVICE)
    
    # 2. Loss and Optimizer
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Simple Dummy Dataset for Demonstration (Replace with actual paths)
    # Using current directory as placeholder, obviously this won't work without real data
    # dataset = RoadDataset("data/train/images", "data/train/masks")
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Training script template ready.")
    print("To train, organize your data into 'images' and 'masks' folders and uncomment the dataloader lines.")
    print("Starting Dummy Training Loop Check...")
    
    # Dummy input to verify loop
    dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    dummy_target = torch.randint(0, 2, (2, 1, IMG_SIZE, IMG_SIZE)).float().to(DEVICE)
    
    model.train()
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    optimizer.step()
    
    print(f"Dummy Batch Loss: {loss.item():.4f}")
    print("Model forward/backward pass successful!")
    
    # Save dummy weights
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved initial weights to {SAVE_PATH}")

if __name__ == "__main__":
    train()
