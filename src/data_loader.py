import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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
            # We can use np.load with mmap_mode='r' to get shape without full load
            # Wait, np.savez doesn't support mmap directly for getting shape without loading the array,
            # but we can load just the metadata. Actually, for a handful of GBs across many files, 
            # loading lazy chunks is possible or we just load shapes.
            data = np.load(file_path)
            num_chunks = data['chunks'].shape[0]
            label = float(data['label'])
            
            for chunk_idx in range(num_chunks):
                self.samples.append({
                    'file': file_path,
                    'chunk_idx': chunk_idx,
                    'label': label
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
        
        # Load file if not in cache (handles sequential loading relatively well if DataLoader workers=0)
        # Note: with multiprocess dataloading, caching logic needs to be careful, but we load full participant file.
        if self.current_cache_file != file_path:
            # Load participant data
            data = np.load(file_path)
            self.current_cache_chunks = data['chunks']
            self.current_cache_file = file_path
            
        chunk_data = self.current_cache_chunks[chunk_idx]
        x = torch.tensor(chunk_data, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)
        
        if self.normalize:
            x = (x - self.mean) / self.std
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
        return x, y

def get_dataloaders(data_dir, batch_size=32, num_workers=0):
    """
    Returns (train_loader, dev_loader)
    Note: num_workers=0 is the safe default on Windows (no fork support).
    """
    train_dataset = DAIC_Dataset(data_dir, split="train")
    # Setting workers > 0 requires care with our naive file cache. 
    # Actually, the file cache is per-worker, so it works perfectly.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
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
