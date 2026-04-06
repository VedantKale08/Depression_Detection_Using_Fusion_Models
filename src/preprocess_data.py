import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def create_global_stats():
    return {
        'count': 0,
        'sum': 0.0,
        'sum_sq': 0.0
    }

def update_global_stats(stats, batch_data):
    # batch_data is (N, frames, features), we can reshape to (N*frames, features)
    flat_data = batch_data.reshape(-1, batch_data.shape[-1])
    n = flat_data.shape[0]
    if n == 0:
        return
    
    stats['count'] += n
    stats['sum'] += flat_data.sum(axis=0)
    stats['sum_sq'] += (flat_data ** 2).sum(axis=0)

def finalize_stats(stats):
    mean = stats['sum'] / stats['count']
    var = (stats['sum_sq'] / stats['count']) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-8)) # Add small epsilon
    return mean, std

def process_participant(participant_id, data_dir, output_dir, chunk_size=300):
    part_dir = os.path.join(data_dir, f"{participant_id}_P")
    
    covarep_path = os.path.join(part_dir, f"{participant_id}_COVAREP.csv")
    clnf_path = os.path.join(part_dir, f"{participant_id}_CLNF_features.txt")
    audio_emo_path = os.path.join(part_dir, f"{participant_id}_audio_emotion.npy")
    text_emo_path = os.path.join(part_dir, f"{participant_id}_text_emotion.npy")
    
    # Check if necessary files exist
    if not (os.path.exists(covarep_path) and os.path.exists(clnf_path)):
        return None
    
    # 1. Load COVAREP
    covarep = pd.read_csv(covarep_path, header=None)
    covarep['timestamp'] = np.arange(len(covarep)) * 0.01
    covarep['time_window'] = np.floor(covarep['timestamp'] * 10) / 10
    covarep_10hz = covarep.drop(columns=['timestamp']).groupby('time_window').mean()
    
    # 2. Load CLNF
    # Note: the separator in the CLNF file appears to be comma and sometimes spaces
    clnf = pd.read_csv(clnf_path, sep=r',\s*', engine='python')
    clnf['time_window'] = np.floor(clnf['timestamp'] * 10) / 10
    clnf_10hz = clnf.drop(columns=['frame', 'timestamp', 'face_id', 'confidence', 'success'], errors='ignore').groupby('time_window').mean()
    
    # Merge
    merged = pd.merge(covarep_10hz, clnf_10hz, left_index=True, right_index=True, how='outer')
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.ffill().fillna(0) # Forward fill, then fill remaining with 0
    
    length = len(merged)
    
    # 3. Load emotions
    audio_emotion = np.load(audio_emo_path) if os.path.exists(audio_emo_path) else np.zeros((7,))
    text_emotion = np.load(text_emo_path) if os.path.exists(text_emo_path) else np.zeros((7,))
    
    # Per-frame features: COVAREP (74) + CLNF (136) = 210
    # Audio/Text emotions are saved separately as auxiliary features — NOT tiled per frame.
    # This prevents the model from memorizing participant identity via static emotion signatures.
    features = np.concatenate([
        merged.iloc[:, :74].values,   # COVAREP (74)
        merged.iloc[:, 74:].values,   # CLNF (136)
    ], axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Chunking
    chunks = []
    for start in range(0, length, chunk_size):
        end = start + chunk_size
        chunk = features[start:end]
        
        # Zero padding if short
        if len(chunk) < chunk_size:
            pad_len = chunk_size - len(chunk)
            chunk = np.pad(chunk, ((0, pad_len), (0, 0)), mode='constant')
            
        chunks.append(chunk)
    
    # Return chunks AND the per-session emotion vectors (7+7=14 dims)
    emotion_vec = np.concatenate([audio_emotion, text_emotion], axis=0).astype(np.float32)  # (14,)
    return np.array(chunks), emotion_vec

def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    # The columns are Participant_ID, PHQ8_Binary, etc.
    return dict(zip(df['Participant_ID'].astype(str), df['PHQ8_Binary']))

def main():
    base_dir = "data/raw/DAIC_WOZ"
    output_dir = "data/processed"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get labels
    train_labels = load_labels(os.path.join(base_dir, "train_split_Depression_AVEC2017.csv"))
    
    dev_labels_path = os.path.join(base_dir, "dev_split_Depression_AVEC2017.csv")
    dev_labels = load_labels(dev_labels_path) if os.path.exists(dev_labels_path) else {}
    
    test_labels_path = os.path.join(base_dir, "full_test_split.csv")
    test_labels = load_labels(test_labels_path) if os.path.exists(test_labels_path) else {}
    
    all_labels = {**train_labels, **dev_labels, **test_labels}
    
    stats = create_global_stats()
    
    part_dirs = [d for d in os.listdir(base_dir) if d.endswith('_P') and os.path.isdir(os.path.join(base_dir, d))]
    
    print(f"Found {len(part_dirs)} participant folders. Processing...")
    
    for i, pd_name in enumerate(part_dirs):
        if i % 10 == 0:
            print(f"Processing folder {i}/{len(part_dirs)}: {pd_name}")
        part_id = pd_name.split('_')[0]
        if part_id not in all_labels:
            print(f"  WARNING: Folder '{pd_name}' (ID={part_id}) not found in label CSVs — skipping.")
            continue
            
        label = all_labels[part_id]
        
        # Determine split
        if part_id in train_labels:
            split = "train"
        elif part_id in dev_labels:
            split = "dev"
        else:
            split = "test"
        split_out_dir = os.path.join(output_dir, split)
        os.makedirs(split_out_dir, exist_ok=True)
        
        result = process_participant(part_id, base_dir, split_out_dir)
        if result is None:
            continue
        chunks, emotion_vec = result
            
        # Only update normalization stats from TRAIN split to prevent data leakage
        if split == "train":
            update_global_stats(stats, chunks)
        
        # Save chunks, emotion vector, and label as npz
        # chunks: (num_chunks, chunk_size, 210) — per-frame temporal features
        # emotion: (14,)                        — session-level auxiliary features
        # label: scalar
        np.savez(os.path.join(split_out_dir, f"{part_id}.npz"),
                 chunks=np.array(chunks, dtype=np.float32),
                 emotion=emotion_vec,
                 label=np.array(label, dtype=np.float32))
    
    # Finalize and save normalization stats
    mean, std = finalize_stats(stats)
    np.savez(os.path.join(output_dir, "normalization_params.npz"),
             mean=np.array(mean, dtype=np.float32), 
             std=np.array(std, dtype=np.float32))
    
    print("Preprocessing completed!")
    print(f"Feature size per frame: {mean.shape[0]}")
    print(f"Stats saved to {os.path.join(output_dir, 'normalization_params.pt')}")

if __name__ == "__main__":
    main()
