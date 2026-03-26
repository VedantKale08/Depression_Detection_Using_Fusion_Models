import pandas as pd
import numpy as np
import os

data_dir = "test_data/DAIC_WOZ/300_P"

# 1. Load COVAREP (100 Hz, 0.01s per frame)
print("Loading COVAREP...")
covarep = pd.read_csv(os.path.join(data_dir, "300_COVAREP.csv"), header=None)
print(f"COVAREP shape: {covarep.shape}")
# Create timestamp for COVAREP: 0, 0.01, 0.02, ...
covarep['timestamp'] = np.arange(len(covarep)) * 0.01

# 2. Load CLNF (30 Hz)
print("Loading CLNF...")
clnf = pd.read_csv(os.path.join(data_dir, "300_CLNF_features.txt"), sep=',\s*', engine='python')
print(f"CLNF shape (raw): {clnf.shape}")
print(f"CLNF columns: {clnf.columns[:5]}")

# 3. Downsample both to 10 Hz (0.1s windows)
# We can create a 'time_window' column which is timestamp rounded down to nearest 0.1
# e.g., floor(timestamp * 10) / 10
covarep['time_window'] = np.floor(covarep['timestamp'] * 10) / 10
clnf['time_window'] = np.floor(clnf['timestamp'] * 10) / 10

covarep_10hz = covarep.drop(columns=['timestamp']).groupby('time_window').mean()
clnf_10hz = clnf.drop(columns=['frame', 'timestamp', 'face_id', 'confidence', 'success'], errors='ignore').groupby('time_window').mean()

print(f"COVAREP 10hz shape: {covarep_10hz.shape}")
print(f"CLNF 10hz shape: {clnf_10hz.shape}")

# Merge them on time_window
# Use outer join to see overlap, then ffill/bfill or fillna(0)
merged = pd.merge(covarep_10hz, clnf_10hz, left_index=True, right_index=True, how='outer')
print(f"Merged shape: {merged.shape}")
print(f"Merged missing COVAREP: {merged.iloc[:, 0].isna().sum()}")
print(f"Merged missing CLNF: {merged.iloc[:, -1].isna().sum()}")

# Fill missing values
merged = merged.ffill().fillna(0)
print(f"Merged memory usage: {merged.memory_usage().sum() / 1024**2:.2f} MB")

# Add static emotion vectors (184-dim each)
audio_emotion = np.load(os.path.join(data_dir, "300_audio_emotion.npy"))
text_emotion = np.load(os.path.join(data_dir, "300_text_emotion.npy"))

print(f"Emotion shapes: Audio={audio_emotion.shape}, Text={text_emotion.shape}")
