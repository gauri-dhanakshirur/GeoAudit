# Phase 1: Training (The Setup)
**Goal**: Teach the AI to see roads.

## ALGORITHM: The Teacher
**Get** raw data from "the World".
**Create** a learning loop (The Epochs).

### Step 1: Preparation (The Materials)
**IF** processing "Raw Data":
    **Action**: Run **Real-ESRGAN**.
    **Result**: 4x Upscaled Images (Recovered textures).
**IF** creating "Knowledge":
    **Action**: Run `label_tool.py`.
    **Result**: Human-verified "Ground Truth" masks.

### Step 2: Observation (The Forward Pass)
**FOR** each Batch of Images:
    **Input**: Patch (256x256).
    **Model Action**:
        **ResNet34**: "I see shapes."
        **D-Block**: "I see the context."
        **Strip Pooling**: "I see lines (roads)."
    **Prediction**: "This is where I think the road is."

### Step 3: Correction (The Backward Pass)
**Measure Error**: Compare "Prediction" vs "Ground Truth" (DiceBCE Loss).
**Feedback**: "You missed this spot." (Backpropagation).
**Improve**: Adjust neuron weights (Optimizer).
**Repeat**: 50 times (Epochs).

---

# Phase 2: Execution (The Application)
**Goal**: Use the AI to analyze new data.

## ALGORITHM: The Explorer
**Get** new image from "the User".
**Create** an analysis pipeline.

### Step 1: Segmentation (The Eyes)
**Input**: Unseen Satellite Image.
**Action**:
    **Divide**: Split into patches.
    **Scan**: Model predicts probability map.
    **Threshold**: IF probability > 50% THEN Pixel = White (Road) ELSE Pixel = Black.

### Step 2: Refinement (The Glasses)
**Action**: **Morphological Closing**.
**Logic**: IF Road -> Gap -> Road THEN Fill Gap.
**Result**: Continuous road network.

### Step 3: Measurement (The Ruler)
**Action**: **Skeletonization**.
**Logic**: Shrink road width to 1 pixel center-line.
**Action**: **Length Calculation**.
**Math**: Count Pixels × Ground Sample Distance (GSD).
**Output**: "Total Road Length: 15.4 km".

### Step 4: Classification (The Geologist)
**Input**: Road Mask + Original Color Image.
**Action**: **Spectral Analysis**.
**Logic**:
    **Identify**: Look at pixels *inside* the road mask.
    **Compare**: Check RGB values.
    **Decision**:
        IF Color is Gray/Black -> **Type: Asphalt**.
        IF Color is Brown/Orange -> **Type: Dirt/Unpaved**.
**Output**: "Road Type: Asphalt".
