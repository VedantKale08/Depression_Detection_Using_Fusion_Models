import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from data_loader import get_dataloaders
from model import DepressionHybridModel
from sklearn.metrics import f1_score, accuracy_score

def train_model(data_dir="data/processed", batch_size=32, epochs=20, learning_rate=1e-3, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare Data Loaders
    train_loader, dev_loader = get_dataloaders(data_dir, batch_size=batch_size, num_workers=4)
    if len(train_loader.dataset) == 0:
        print("Error: Train dataset is empty. Please run preprocess_data.py first.")
        return
        
    print(f"Data Loaded! Train subsets: {len(train_loader.dataset)} | Dev subsets: {len(dev_loader.dataset)}")
    
    # 2. Setup Model
    model = DepressionHybridModel(input_size=224, hidden_size=128, num_layers=2)
    model.to(device)
    
    # 3. Setup Loss and Optimizer
    # We use BCEWithLogitsLoss because it is numerically more robust than Sigmoid + BCELoss
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    best_f1 = 0.0
    os.makedirs("weights", exist_ok=True)
    
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
            
        print(f"Epoch {epoch} Summary:")
        print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        if len(dev_loader) > 0:
            print(f"  Dev   -> Loss: {dev_loss:.4f} | Acc: {dev_acc:.4f} | F1: {dev_f1:.4f}")
            
        # 6. Save Best Weights
        # Save model if F1 score improved on the validation set
        if len(dev_loader) > 0 and dev_f1 >= best_f1 and dev_f1 > 0:
            best_f1 = dev_f1
            torch.save(model.state_dict(), "weights/best_hybrid_model.pth")
            print(f"  --> Saved new best model to 'weights/best_hybrid_model.pth' (F1={best_f1:.4f})")
        # Fallback: Save if no dev set and train F1 improves
        elif len(dev_loader) == 0 and train_f1 >= best_f1:
            best_f1 = train_f1
            torch.save(model.state_dict(), "weights/best_hybrid_model.pth")
            print(f"  --> Saved model to 'weights/best_hybrid_model.pth' (Train F1={best_f1:.4f})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the DAIC-WOZ Hybrid Depression Model")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing processed chunks")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    
    train_model(data_dir=args.data_dir, batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.lr)
