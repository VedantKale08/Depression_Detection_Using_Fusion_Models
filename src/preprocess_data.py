import os
import shutil
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
    au_path = os.path.join(part_dir, f"{participant_id}_CLNF_AUs.txt")
    pose_path = os.path.join(part_dir, f"{participant_id}_CLNF_pose.txt")
    gaze_path = os.path.join(part_dir, f"{participant_id}_CLNF_gaze.txt")
    audio_emo_path = os.path.join(part_dir, f"{participant_id}_audio_emotion.npy")
    text_emo_path = os.path.join(part_dir, f"{participant_id}_text_emotion.npy")
    
    # Audio features are strict requirement
    if not os.path.exists(covarep_path):
        return None
        
    def load_and_group(path):
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, sep=r',\s*', engine='python')
            df.columns = df.columns.str.strip()
            df['time_window'] = np.floor(df['timestamp'] * 10) / 10
            return df.drop(columns=['frame', 'timestamp', 'face_id', 'confidence', 'success'], errors='ignore').groupby('time_window').mean()
        except:
            return pd.DataFrame()
            
    # 1. Load COVAREP (acts as time master)
    covarep = pd.read_csv(covarep_path, header=None)
    covarep['timestamp'] = np.arange(len(covarep)) * 0.01
    covarep['time_window'] = np.floor(covarep['timestamp'] * 10) / 10
    merged = covarep.drop(columns=['timestamp']).groupby('time_window').mean()
    
    # 2. Load Face Features
    au_10hz = load_and_group(au_path)
    pose_10hz = load_and_group(pose_path)
    gaze_10hz = load_and_group(gaze_path)
    
    # Join into one Master DataFrame aligned by COVAREP time
    face_df = pd.DataFrame(index=merged.index)
    if not au_10hz.empty:
        face_df = face_df.join(au_10hz, how='left')
    if not pose_10hz.empty:
        face_df = face_df.join(pose_10hz, how='left')
    if not gaze_10hz.empty:
        face_df = face_df.join(gaze_10hz, how='left')
        
    face_df = face_df.ffill().fillna(0) # Forward fill, then fill remaining with 0
    
    length = len(merged)
    
    # 3. Load emotions (Session Level)
    audio_emotion = np.load(audio_emo_path) if os.path.exists(audio_emo_path) else np.zeros((7,))
    text_emotion = np.load(text_emo_path) if os.path.exists(text_emo_path) else np.zeros((7,))
    
    # Prepare standard dimensions
    au_cols = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r', 'AU04_c', 'AU12_c', 'AU15_c', 'AU23_c', 'AU28_c', 'AU45_c']
    pose_cols = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
    gaze_cols = ['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'x_h0', 'y_h0', 'z_h0', 'x_h1', 'y_h1', 'z_h1']
    
    au_vals = face_df.reindex(columns=au_cols, fill_value=0.0).values
    pose_vals = face_df.reindex(columns=pose_cols, fill_value=0.0).values
    gaze_vals = face_df.reindex(columns=gaze_cols, fill_value=0.0).values
    
    # Per-frame features: COVAREP (74) + AUs (20) + Pose (6) + Gaze (12) = 112
    features = np.concatenate([
        merged.iloc[:, :74].values,   # COVAREP (74)
        au_vals,                      # AUs (20)
        pose_vals,                    # Pose (6)
        gaze_vals                     # Gaze (12)
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
    emotion_vec = np.concatenate([audio_emotion, text_emotion], axis=0).astype(np.float32)
    return np.array(chunks), emotion_vec

def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    # The columns are Participant_ID, PHQ8_Binary, etc.
    return dict(zip(df['Participant_ID'].astype(str), df['PHQ8_Binary']))

def main():
    base_dir = "data/raw/DAIC_WOZ"
    output_dir = "data/processed"
    
    # Clean up any old processed split files before regenerating new 112-D data
    if os.path.exists(output_dir):
        for subdir in ["train", "dev", "test"]:
            path = os.path.join(output_dir, subdir)
            if os.path.exists(path):
                shutil.rmtree(path)
        norm_path = os.path.join(output_dir, "normalization_params.npz")
        if os.path.exists(norm_path):
            os.remove(norm_path)
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
        # chunks: (num_chunks, chunk_size, 112) — per-frame temporal features
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