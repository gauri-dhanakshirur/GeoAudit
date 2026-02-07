# Algorithm: The Road Extraction Pipeline
**Goal**: Transform raw satellite data into actionable road network methodology.

## ALGORITHM: The Pipeline Director
**Get** the Lifecycle Stages from "The Developer".
**Create** a strict execution sequence (Training -> Execution).
**FOR** each Phase in the "Lifecycle": Create a "Processing Block":

    ### IF the Phase is "Training" (The Setup):
        **Set status** to "Learning Mode".
        **Define layout** as "Iterative Improvement".
        
        **IF** the step is "Data Ingestion":
            **Input**: Low-Res Satellite Tiles.
            **Action**: Feed into **Real-ESRGAN** Generator.
            **Output**: 4x Upscaled Images (Recovered Textures).
            
        **IF** the step is "Annotation":
            **Input**: Upscaled Images.
            **Tool**: `label_tool.py` (OpenCV).
            **Action**: Human annotator draws "Vectors" (White Lines).
            **Output**: Binary Ground Truth Masks.
            
        **IF** the step is "Model Optimization":
            **Model**: **AdvancedRoadModel** (ResNet34 + D-LinkNet + Attention).
            **Loop**: 50 Epochs.
            **Logic**: 
                Calculate **DiceBCE Loss** (Compare Prediction vs Truth).
                Backpropagate Error.
                Update Weights.

    ### IF the Phase is "Execution" (The Application):
        **Set status** to "Inference Mode".
        **Define layout** as "Real-time Analysis".
        
        **IF** the User Input is a "New Image":
            **Set Context**: Unseen Satellite Data.
            **Action**: Pre-process (Resize/Patching).
            
        **IF** the Action is "Segmentation":
            **Execute**: Forward Pass through Model.
            **Mechanism**: 
                **ResNet34**: Extracts features.
                **Strip Pooling**: Captures long vertical/horizontal lines (Roads).
                **Attention Gates**: Filters out forests/buildings.
            **Output**: Probability Probability Map (0.0 - 1.0).
            **Threshold**: Convert > 0.5 to White (Road).

        **IF** the Action is "Analysis":
            **Refine**: Apply **Morphological Closing** (Heal occlusions/gaps).
            **Thin**: Apply **Skeletonization** (Reduce to 1-pixel centerlines).
            **Measure**: Count pixels * GSD (Ground Sample Distance).
            **Result**: Total Road Length in Kilometers.
