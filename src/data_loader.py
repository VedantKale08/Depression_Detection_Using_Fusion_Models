import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class DAIC_Dataset(Dataset):
    def __init__(self, data_dir, split="train", normalize=True):
        """
        Args:
            data_dir (str): Path to the processed_data directory.
            split (str): "train" or "dev" (or "test").
            normalize (bool): Whether to apply Z-score normalization.
        """
        self.data_dir = data_dir
        self.split = split
        self.split_dir = os.path.join(data_dir, split)
        self.normalize = normalize
        
        # Load normalization params if needed
        if self.normalize:
            params_path = os.path.join(data_dir, "normalization_params.npz")
            if os.path.exists(params_path):
                params = np.load(params_path)
                self.mean = torch.tensor(params['mean'], dtype=torch.float32)
                self.std = torch.tensor(params['std'], dtype=torch.float32)
            else:
                print(f"Warning: Normalization params not found at {params_path}.")
                self.normalize = False
                
        # We index all chunks. Each .npz file has shape (num_chunks, chunk_size, num_features)
        # To avoid loading everything into RAM, we first build an index:
        # File paths -> number of chunks inside
        self.files = glob.glob(os.path.join(self.split_dir, "*.npz"))
        self.samples = []
        
        print(f"Building index for {split} split...")
        for file_path in self.files:
            data = np.load(file_path)
            num_chunks = data['chunks'].shape[0]
            label = float(data['label'])
            # Load session-level emotion vector (14,): audio(7) + text(7)
            emotion_vec = data['emotion'] if 'emotion' in data else np.zeros(14, dtype=np.float32)
            
            for chunk_idx in range(num_chunks):
                self.samples.append({
                    'file': file_path,
                    'chunk_idx': chunk_idx,
                    'label': label,
                    'emotion': emotion_vec   # shared across all chunks of this participant
                })
                
        print(f"Found {len(self.samples)} chunks in {len(self.files)} participants.")

        
        # For efficiency, we will cache the last loaded file
        self.current_cache_file = None
        self.current_cache_chunks = None
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        file_path = sample_info['file']
        chunk_idx = sample_info['chunk_idx']
        label = sample_info['label']
        emotion_vec = sample_info['emotion']  # (14,) — audio(7) + text(7)
        
        # Load file if not in cache
        if self.current_cache_file != file_path:
            data = np.load(file_path)
            self.current_cache_chunks = data['chunks']
            self.current_cache_file = file_path
            
        chunk_data = self.current_cache_chunks[chunk_idx]
        x = torch.tensor(chunk_data, dtype=torch.float32)          # (300, 210)
        emotion = torch.tensor(emotion_vec, dtype=torch.float32)   # (14,)
        y = torch.tensor(label, dtype=torch.float32)
        
        # Normalize only the temporal sequence (NOT the emotion vector)
        if self.normalize:
            x = (x - self.mean) / self.std
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Add random Gaussian noise to temporal features during training
        if self.split == "train":
            noise = torch.randn_like(x) * 0.05
            x = x + noise
            
        return x, emotion, y

def get_dataloaders(data_dir, batch_size=32, num_workers=0):
    """
    Returns (train_loader, dev_loader)
    Note: num_workers=0 is the safe default on Windows (no fork support).
    """
    train_dataset = DAIC_Dataset(data_dir, split="train")
    
    # Calculate weights for Balanced WeightedRandomSampler
    labels = [sample['label'] for sample in train_dataset.samples]
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    
    weight_neg = 1.0 / max(neg_count, 1)
    weight_pos = 1.0 / max(pos_count, 1)
    sample_weights = [weight_pos if l == 1 else weight_neg for l in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Shuffle must be False when using sampler
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dev_dataset = DAIC_Dataset(data_dir, split="dev")
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, dev_loader

def get_test_dataloader(data_dir, batch_size=32, num_workers=0):
    """
    Returns test_loader
    """
    test_dataset = DAIC_Dataset(data_dir, split="test")
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    return test_loader

if __name__ == "__main__":
    # Test script locally
    # It assumes the user runs this in their PyTorch environment
    print("Testing DAIC_Dataset...")
    try:
        train_loader, dev_loader = get_dataloaders("data/processed", batch_size=4, num_workers=0)
        test_loader = get_test_dataloader("data/processed", batch_size=4, num_workers=0)
        for x, y in train_loader:
            print(f"Batch X shape: {x.shape}")
            print(f"Batch Y shape: {y.shape}")
            print(f"First element mean after norm: {x[0].mean().item():.4f}")
            break
        print("DataLoader test successful!")
    except Exception as e:
        print(f"Test failed or PyTorch is not available: {e}")
