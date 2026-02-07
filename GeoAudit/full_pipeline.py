import os
import subprocess
import argparse
import sys
import glob
from label_tool import label_images
from train import train

def run_super_resolution(input_dir, output_dir):
    """
    Runs Real-ESRGAN to upscale images from input_dir to output_dir.
    """
    print("\n--- STEP 1: SUPER RESOLUTION (Real-ESRGAN) ---")
    
    # Check if Real-ESRGAN script exists
    script_path = os.path.join("Real_ESRGAN", "inference_realesrgan.py")
    if not os.path.exists(script_path):
        print(f"Error: Could not find {script_path}")
        print("Please make sure the 'Real_ESRGAN' folder is in the current directory.")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct command
    # python Real_ESRGAN/inference_realesrgan.py -i <input> -o <output> -n RealESRGAN_x4plus --outscale 4 --face_enhance
    # We omit --face_enhance for roads usually, but usage showed it in args. 
    # For satellite imagery, strictly speaking face enhance is irrelevant/bad. 
    # Let's stick to standard x4plus.
    
    cmd = [
        sys.executable, script_path,
        "-i", input_dir,
        "-o", output_dir,
        "-n", "RealESRGAN_x4plus",
        "--outscale", "4",
        "--suffix", "" # No suffix so filenames remain easy to track
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("Super Resolution Completed Successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running Real-ESRGAN: {e}")
        return False

def run_labeling(input_dir, train_img_dir, train_mask_dir):
    """
    Runs the labeling tool on the upscaled images.
    """
    print("\n--- STEP 2: DATA LABELING ---")
    print(f"Input Images: {input_dir}")
    print(f"Saving Labels to: {train_mask_dir}")
    
    # We call the function directly since we imported it
    # But label_tool.py might expect to run as main if we didn't refactor it to expose a function.
    # checking label_tool.py... it has `def label_images(...)` and `if __name__ == '__main__':`
    # So we can call label_images directly.
    
    try:
        label_images(input_dir, train_img_dir, train_mask_dir)
        print("Labeling Session Finished.")
    except Exception as e:
        print(f"Error during labeling: {e}")

def run_training_prompt():
    """
    Asks user if they want to train the model.
    """
    print("\n--- STEP 3: MODEL TRAINING ---")
    choice = input("Do you want to start training the model now? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("Starting Training...")
        try:
            train() # Call the train function from train.py
        except Exception as e:
            print(f"Error during training: {e}")
    else:
        print("Skipping training. You can run 'python train.py' later.")

def main():
    parser = argparse.ArgumentParser(description="Full End-to-End Pipeline: SR -> Label -> Train")
    parser.add_argument("--raw_input", default="raw_data", help="Folder with raw low-res images")
    parser.add_argument("--sr_output", default="high_res_data", help="Folder for upscaled images")
    parser.add_argument("--train_images", default="data/train/images", help="Final training images folder")
    parser.add_argument("--train_masks", default="data/train/masks", help="Final training masks folder")
    
    args = parser.parse_args()
    
    # 0. Check Input
    if not os.path.exists(args.raw_input):
        print(f"Creating Input Directory: {args.raw_input}")
        os.makedirs(args.raw_input)
        print(f"Please put your raw satellite images in '{args.raw_input}' and rerun the script.")
        return

    # Check if empty
    if not glob.glob(os.path.join(args.raw_input, "*")):
        print(f"'{args.raw_input}' is empty. Please add some images.")
        return

    # 1. Super Resolution
    success = run_super_resolution(args.raw_input, args.sr_output)
    if not success:
        print("Aborting pipeline due to SR failure.")
        return
        
    # 2. Labeling
    run_labeling(args.sr_output, args.train_images, args.train_masks)
    
    # 3. Training
    run_training_prompt()

if __name__ == "__main__":
    main()
