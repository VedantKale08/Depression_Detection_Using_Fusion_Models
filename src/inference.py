import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# Add project root and src to python path to resolve local imports cleanly
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from extract_raw_features import process_video
from preprocess_data import process_participant
from model import DepressionHybridModel

def predict(video_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Names and Paths Setup
    participant_id = os.path.splitext(os.path.basename(video_path))[0]
    data_dir = "data/input"
    out_dir = os.path.join(data_dir, f"{participant_id}_P")
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Extract Raw Features (OpenFace, Whisper, Parselmouth, Emotion Extractors)
    print("\n" + "="*60)
    print("STEP 1: Extracting raw features from video")
    print("="*60)
    process_video(video_path, participant_id)
    
    # 3. Preprocess and Window Features (Merge into 224 features per frame)
    print("\n" + "="*60)
    print("STEP 2: Windowing features into 30-second sequences")
    print("="*60)
    
    # process_participant expects the parent directory of {id}_P
    # It returns an array of shape (num_chunks, 300, 224)
    chunks = process_participant(participant_id, data_dir, data_dir, chunk_size=300)
    
    if chunks is None or len(chunks) == 0:
        print("Error: Could not extract features and chunk them correctly.")
        print("Please check if the required files (.csv, .txt, .npy) were generated successfully in data/input/")
        return
        
    print(f"Successfully generated {len(chunks)} sequence(s).")
    
    # 4. Normalize the Features
    print("\n" + "="*60)
    print("STEP 3: Normalizing features")
    print("="*60)
    norm_path = "data/processed/normalization_params.npz"
    if not os.path.exists(norm_path):
        print(f"Error: Normalization parameters not found at {norm_path}")
        print("Please ensure you have trained the model or run preprocess_data.py on your DAIC_WOZ dataset first.")
        return
        
    params = np.load(norm_path)
    mean = params['mean']
    std = params['std']
    
    # Apply Z-score Normalization (same as in data_loader.py)
    chunks = (chunks - mean) / std
    
    x_tensor = torch.tensor(chunks, dtype=torch.float32).to(device)
    
    # 5. Load Model and Predict
    print("\n" + "="*60)
    print("STEP 4: Running Hybrid Depression Model")
    print("="*60)
    
    model_path = "weights/best_hybrid_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights not found at {model_path}.")
        return
        
    # Initialize Multimodal Sequence Model
    model = DepressionHybridModel(input_size=224, hidden_size=128, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        
    print("\n" + "="*60)
    print("FINAL INFERENCE RESULTS")
    print("="*60)
    
    # Aggregate predictions across chunks by averaging probability
    avg_prob = np.mean(probs)
    
    print(f"Analyzed {len(probs)} valid segment(s) from the video.")
    for i, p in enumerate(probs):
        print(f"  Sequence {i+1} Probability: {p:.4f} ({(p*100):.1f}%)")
        
    print("\n------------------------------------------------------------")
    print(f"OVERALL DEPRESSION PROBABILITY: {avg_prob:.4f} ({(avg_prob*100):.1f}%)")
    
    if avg_prob >= 0.5:
        print("DIAGNOSIS: The model predicts a HIGH likelihood of Depression.")
    else:
        print("DIAGNOSIS: The model predicts a LOW likelihood of Depression.")
    print("------------------------------------------------------------\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Depression Detection Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (.mp4)")
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Input video not found at '{args.video}'")
        sys.exit(1)
        
    predict(args.video)
