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


def aggregate_sequence_probs(probs):
    """Return a robust summary across sequence probabilities."""
    probs = np.asarray(probs).reshape(-1)
    mean_prob = float(np.mean(probs))
    median_prob = float(np.median(probs))
    max_prob = float(np.max(probs))
    min_prob = float(np.min(probs))

    buckets = {
        ">= 0.70": probs >= 0.70,
        "0.50 - 0.70": (probs >= 0.50) & (probs < 0.70),
        "0.30 - 0.50": (probs >= 0.30) & (probs < 0.50),
        "< 0.30": probs < 0.30,
    }
    counts = {label: int(mask.sum()) for label, mask in buckets.items()}
    percentages = {label: float(mask.mean()) for label, mask in buckets.items()}
    positive_pct = float((probs >= 0.50).mean())
    high_pct = float((probs >= 0.70).mean())

    depression_level = categorize_depression_level(median_prob)

    return {
        "mean_prob": mean_prob,
        "median_prob": median_prob,
        "max_prob": max_prob,
        "min_prob": min_prob,
        "counts": counts,
        "percentages": percentages,
        "positive_pct": positive_pct,
        "high_pct": high_pct,
        "depression_level": depression_level,
    }


def categorize_depression_level(probability):
    """Map a probability value to low/medium/high depression categories."""
    if probability > 0.65:
        return "high"
    if probability > 0.35:
        return "medium"
    return "low"


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
    
    # 3. Preprocess and Window Features (Merge into 210 features per frame)
    print("\n" + "="*60)
    print("STEP 2: Windowing features into 30-second sequences")
    print("="*60)
    
    # process_participant expects the parent directory of {id}_P
    # It returns chunks (num_chunks, 300, 210) and emotion_vec (14,)
    result = process_participant(participant_id, data_dir, data_dir, chunk_size=300)
    
    if result is None:
        print("Error: Could not extract features and chunk them correctly.")
        print("Please check if the required files (.csv, .txt, .npy) were generated successfully in data/input/")
        return
        
    chunks, emotion_vec = result
        
    if len(chunks) == 0:
        print("Error: Extracted chunks are empty.")
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
    
    # Prepare emotion tensor (replicate emotion_vec for each chunk in the batch)
    emotion_expanded = np.tile(emotion_vec, (len(chunks), 1))
    emo_tensor = torch.tensor(emotion_expanded, dtype=torch.float32).to(device)
    
    # 5. Load Model and Predict
    print("\n" + "="*60)
    print("STEP 4: Running Hybrid Depression Model")
    print("="*60)
    
    model_path = "weights/best_hybrid_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights not found at {model_path}.")
        return
        
    # Initialize Multimodal Sequence Model
    model = DepressionHybridModel(input_size=210, hidden_size=64, num_layers=1)
    
    import warnings
    # Suppress the PyTorch warning about weights_only default changing since we trust the local model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])    
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        logits = model(x_tensor, emo_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        
    summary = aggregate_sequence_probs(probs)

    print("\n" + "="*60)
    print("FINAL INFERENCE RESULTS")
    print("="*60)
    
    print(f"Analyzed {len(probs)} valid segment(s) from the video.")
    for i, p in enumerate(probs):
        print(f"  Sequence {i+1} Probability: {p:.4f} ({(p*100):.1f}%)")
        
    level = categorize_depression_level(summary['median_prob'])
    print("\n------------------------------------------------------------")
    print(f"Depression level:   {level.upper()} ({(summary['median_prob']*100):.1f}%)")
    print(f"Sequences >= 50%:  {summary['positive_pct']*100:.1f}%")
    print(f"Sequences >= 70%:  {summary['high_pct']*100:.1f}%")
    
    print("\nSegment bucket distribution:")
    for label in [">= 0.70", "0.50 - 0.70", "0.30 - 0.50", "< 0.30"]:
        count = summary['counts'][label]
        pct = summary['percentages'][label] * 100
        print(f"  {label}: {count} segment(s) ({pct:.1f}%)")
    
    print("\n------------------------------------------------------------")
    if level == "high":
        print("DIAGNOSIS: HIGH depression risk — must immediately consult a doctor.")
    elif level == "medium":
        print("DIAGNOSIS: MEDIUM depression risk — user should pay attention.")
    else:
        print("DIAGNOSIS: LOW depression risk — can be ignored.")
    print("------------------------------------------------------------\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Depression Detection Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (.mp4)")
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Input video not found at '{args.video}'")
        sys.exit(1)
        
    predict(args.video)
