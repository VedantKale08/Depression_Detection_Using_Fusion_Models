import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from pathlib import Path

# Add project root to python path to resolve data_loader and model imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add src to path if running from root so it finds data_loader and model
src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tqdm import tqdm
from data_loader import get_dataloaders
from model import DepressionHybridModel
from sklearn.metrics import f1_score, accuracy_score

def train_model(data_dir="data/processed", batch_size=32, epochs=20, learning_rate=1e-3, device=None, resume=True):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare Data Loaders
    train_loader, dev_loader = get_dataloaders(data_dir, batch_size=batch_size, num_workers=0)
    if len(train_loader.dataset) == 0:
        print("Error: Train dataset is empty. Please run preprocess_data.py first.")
        return
        
    print(f"Data Loaded! Train subsets: {len(train_loader.dataset)} | Dev subsets: {len(dev_loader.dataset)}")
    
    # 2. Setup Model
    model = DepressionHybridModel(input_size=224, hidden_size=128, num_layers=2)
    model.to(device)
    
    # 3. Setup Loss and Optimizer
    # Compute pos_weight to counteract class imbalance (depressed vs. not-depressed)
    # pos_weight = (# negative samples) / (# positive samples)
    pos_count = sum(s['label'] for s in train_loader.dataset.samples)
    neg_count = len(train_loader.dataset.samples) - pos_count
    pos_weight_val = neg_count / max(pos_count, 1)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    print(f"Class balance — Depressed chunks: {int(pos_count)} | Non-depressed chunks: {int(neg_count)} | pos_weight: {pos_weight_val:.2f}")
    
    # BCEWithLogitsLoss with pos_weight is numerically stable and handles imbalanced datasets
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # ReduceLROnPlateau: halve LR if dev F1 doesn't improve for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_f1 = 0.0
    os.makedirs("weights", exist_ok=True)
    
    # 4. Resume from checkpoint if available
    checkpoint_path = "weights/best_hybrid_model.pth"
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_f1 = checkpoint.get('best_f1', 0.0)
            print(f"Resumed from checkpoint '{checkpoint_path}' (best F1 so far: {best_f1:.4f})")
        else:
            # Legacy: checkpoint is just model state_dict
            model.load_state_dict(checkpoint)
            print(f"Resumed model weights from '{checkpoint_path}' (no optimizer state found)")
    else:
        print("No checkpoint found — training from scratch.")

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for x_batch, y_batch in loop:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Forward
            optimizer.zero_grad()
            logits = model(x_batch)
            
            # Loss
            loss = criterion(logits, y_batch)
            train_loss += loss.item()
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # Prevent exploding gradients
            optimizer.step()
            
            # Predictions (Sigmoid > 0.5 is equivalent to Logits > 0.0)
            preds = (logits > 0.0).float()
            
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(y_batch.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())
            
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, zero_division=0)
        
        # 5. Validation Loop
        model.eval()
        dev_loss = 0.0
        dev_preds, dev_targets = [], []
        
        if len(dev_loader) > 0:
            with torch.no_grad():
                for x_batch, y_batch in tqdm(dev_loader, desc=f"Epoch {epoch}/{epochs} [Dev]"):
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
                    
                    dev_loss += loss.item()
                    preds = (logits > 0.0).float()
                    
                    dev_preds.extend(preds.cpu().numpy())
                    dev_targets.extend(y_batch.cpu().numpy())
                    
            dev_loss /= len(dev_loader)
            dev_acc = accuracy_score(dev_targets, dev_preds)
            dev_f1 = f1_score(dev_targets, dev_preds, zero_division=0)
        else:
            dev_loss, dev_acc, dev_f1 = float('inf'), 0.0, 0.0
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} Summary (lr={current_lr:.2e}):")
        print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        if len(dev_loader) > 0:
            print(f"  Dev   -> Loss: {dev_loss:.4f} | Acc: {dev_acc:.4f} | F1: {dev_f1:.4f}")
            # Step scheduler based on validation F1
            scheduler.step(dev_f1)
            
        # 6. Save Best Weights (model + optimizer state for full resume)
        # Save model if F1 score improved on the validation set
        if len(dev_loader) > 0 and dev_f1 >= best_f1 and dev_f1 > 0:
            best_f1 = dev_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, checkpoint_path)
            print(f"  --> Saved new best model to '{checkpoint_path}' (F1={best_f1:.4f})")
        # Fallback: Save if no dev set and train F1 improves
        elif len(dev_loader) == 0 and train_f1 >= best_f1:
            best_f1 = train_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, checkpoint_path)
            print(f"  --> Saved model to '{checkpoint_path}' (Train F1={best_f1:.4f})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the DAIC-WOZ Hybrid Depression Model")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing processed chunks")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--no-resume", action="store_true", help="Start training from scratch, ignoring any saved checkpoint")
    args = parser.parse_args()
    
    train_model(data_dir=args.data_dir, batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.lr, resume=not args.no_resume)
