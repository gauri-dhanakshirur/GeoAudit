# Project Workflow & Algorithms Guide

This document outlines the end-to-end process of your Satellite Road Extraction project, detailing every step from raw data preparation to final model execution.

## Phase 1: The Training Process
*Goal: To teach the AI model how to recognize roads in satellite imagery.*

### Step 1: Data Preparation & Enhancement
**Input**: Low-resolution raw satellite images.
1.  **Ingestion**: Raw images are placed in the `raw_data` folder.
2.  **Super-Resolution (Algorithm: Real-ESRGAN)**:
    *   The script `full_pipeline.py` triggers `inference_realesrgan.py`.
    *   **What happens**: The images are upscaled by **4x** to recover missing details.
    *   **Algorithm**: Uses a **Generative Adversarial Network (GAN)** with a **ResNet-in-ResNet (RRDB)** generator. It "hallucinates" realistic textures based on learned patterns from high-quality photography.

### Step 2: Annotation (Labeling)
**Input**: High-resolution (upscaled) images.
1.  **Tool**: You run `label_tool.py`.
2.  **Process**:
    *   The tool opens images sequentially.
    *   **Human-in-the-Loop**: A human expert (you) draws lines over the roads using the mouse.
    *   **Output**: The tool generates a "Mask" (Binary Image) where:
        *   **White Pixels (1)** = Road
        *   **Black Pixels (0)** = Background
    *   This creates the "Ground Truth" for the AI to learn from.

### Step 3: Model Training
**Input**: Pairs of Images and their corresponding Masks.
1.  **Script**: `train.py`.
2.  **Preprocessing**:
    *   Images are resized to `256x256` patches.
    *   Pixel values are normalized (0 to 1 range).
3.  **Forward Pass**:
    *   The **AdvancedRoadModel** (Custom U-Net) takes the image and predicts a mask.
    *   **Algorithm**:
        *   **ResNet-34 Feature Extraction**: Analyzes textures and shapes.
        *   **Strip Pooling**: Scans for long, continuous lines (roads).
        *   **Dilated Convolutions (D-Block)**: Expands context to see surroundings.
4.  **Loss Calculation (Algorithm: DiceBCE Loss)**:
    *   The model compares its *Prediction* vs. the *Ground Truth*.
    *   It uses **Dice Loss** (measures overlap) and **Binary Cross Entropy** (measures pixel accuracy).
5.  **Backpropagation (Optimization)**:
    *   **Algorithm**: **Adam Optimizer**.
    *   The "error" is sent backwards through the network to adjust the weights, making the model slightly smarter.
    *   This repeats for roughly 50 epochs (cycles).

---

## Phase 2: The Execution Process (Inference)
*Goal: To take a NEW, unseen image and automatically detect roads and calculate their length.*

### Step 1: User Input
*   **Action**: A user provides a new satellite image (e.g., `test_image.png`).

### Step 2: Preprocessing
*   **Algorithm**: **Patching / Sliding Window**.
*   The image might be too large for the GPU memory. The system chops it into smaller squares (e.g., 256x256) or pads it to fit specific dimensions.

### Step 3: AI Inference (Segmentation)
*   **Model**: The trained `AdvancedRoadModel` is loaded.
*   **Process**:
    *   The model looks at the image patch.
    *   **Attention Gates** activate to focus specifically on road-like textures.
    *   The model outputs a "Probability Map": every pixel gets a score from 0.0 (not road) to 1.0 (definitely road).
*   **Thresholding**: Probabilities > 0.5 are converted to pure White (Road); the rest becomes Black.

### Step 4: Post-Processing & Analysis
*   **Goal**: Clean up the messy AI output and get useful numbers.
1.  **Occlusion Cleaning (Morphology)**:
    *   **Algorithm**: **Morphological Closing**.
    *   If the AI detected a road, then a gap (tree), then a road, this step bridges the gap to make the road continuous.
2.  **Skeletonization**:
    *   **Algorithm**: **Thinning (e.g., Zhang-Suen)**.
    *   The thick white road blob is whittled down to a single-pixel-wide line (the "skeleton").
3.  **Length Calculation**:
    *   **Algorithm**: **Graph Traversal / Pixel Counting**.
    *   The total number of pixels in the skeleton is counted.
    *   Using the **GSD (Ground Sample Distance)**, pixels are converted to kilometers (e.g., 1 pixel = 0.3 meters).

### Step 5: Visualization
*   **Output**: The system displays:
    1.  original Image.
    2.  The Detected Binary Mask.
    3.  The Final Analysis (Road length overlaid on the map).
