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
import torch.nn.functional as F
from data_loader import get_dataloaders, get_test_dataloader
from model import DepressionHybridModel, DepressionTransformerModel
from sklearn.metrics import f1_score, accuracy_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

def train_model(data_dir="data/processed", batch_size=32, epochs=20, learning_rate=1e-3, device=None, resume=True, model_type="bi-lstm", hidden_size=16):
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
    # input_size=210: COVAREP(74) + CLNF(136) — emotions are auxiliary inputs at the FC head
    if model_type == "transformer":
        model = DepressionTransformerModel(input_size=210, d_model=128, nhead=4, num_layers=2, dropout=0.5)
        checkpoint_path = "weights/best_transformer_model.pth"
    else:
        # Reduced hidden_size from 64 to the passed parameter (default 16) to combat overfitting
        model = DepressionHybridModel(input_size=210, hidden_size=hidden_size, num_layers=1, dropout=0.6)
        checkpoint_path = f"weights/best_hybrid_model_hs{hidden_size}.pth"
    model.to(device)
    
    # 3. Setup Loss and Optimizer
    # Replaced BCEWithLogitsLoss with FocalLoss to handle depressed vs non-depressed imbalance
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    # ReduceLROnPlateau: halve LR if dev F1 doesn't improve for 4 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    best_f1 = 0.0
    patience_counter = 0
    early_stop_patience = 10  # Stop if dev F1 doesn't improve for 10 epochs
    os.makedirs("weights", exist_ok=True)
    
    # 4. Resume from checkpoint if available
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
        for x_batch, emo_batch, y_batch in loop:
            x_batch   = x_batch.to(device)
            emo_batch = emo_batch.to(device)
            y_batch   = y_batch.to(device)
            
            # Forward
            optimizer.zero_grad()
            logits = model(x_batch, emo_batch)
            
            # Loss
            loss = criterion(logits, y_batch)
            train_loss += loss.item()
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            # Predictions (logits > 0.0 is equivalent to sigmoid > 0.5)
            preds = (logits > 0.0).float()
            
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(y_batch.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())
            
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, zero_division=0)
        
        # 5. Validation Loop — PARTICIPANT-LEVEL aggregation
        # Each .npz file = one participant. We average logits over all their chunks,
        # then threshold once. This is how DAIC-WOZ should be evaluated.
        model.eval()
        dev_loss = 0.0
        dev_preds, dev_targets = [], []
        
        if len(dev_loader) > 0:
            # Build participant -> {logits, label} map
            participant_logits = {}  # key: file_path, value: list of logit values
            participant_labels = {}
            
            with torch.no_grad():
                for batch_idx, (x_batch, emo_batch, y_batch) in enumerate(tqdm(dev_loader, desc=f"Epoch {epoch}/{epochs} [Dev]")):
                    x_batch   = x_batch.to(device)
                    emo_batch = emo_batch.to(device)
                    y_batch   = y_batch.to(device)
                    logits = model(x_batch, emo_batch)
                    loss = criterion(logits, y_batch)
                    dev_loss += loss.item()
                    
                    # Track chunk-level logits per participant using dataset index
                    # batch_idx * batch_size gives starting sample index
                    batch_size_actual = x_batch.shape[0]
                    start_idx = batch_idx * dev_loader.batch_size
                    logits_np = logits.cpu().numpy()
                    labels_np = y_batch.cpu().numpy()
                    
                    for i in range(batch_size_actual):
                        sample_idx = start_idx + i
                        if sample_idx >= len(dev_loader.dataset.samples):
                            break
                        file_path = dev_loader.dataset.samples[sample_idx]['file']
                        if file_path not in participant_logits:
                            participant_logits[file_path] = []
                            participant_labels[file_path] = labels_np[i]
                        participant_logits[file_path].append(logits_np[i])
            
            # Aggregate: mean logit per participant, then threshold
            for fp in participant_logits:
                avg_logit = np.mean(participant_logits[fp])
                pred = 1.0 if avg_logit > 0.0 else 0.0
                dev_preds.append(pred)
                dev_targets.append(participant_labels[fp])
                    
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
        # Save model if participant-level F1 improved on the validation set
        if len(dev_loader) > 0 and dev_f1 >= best_f1 and dev_f1 > 0:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, checkpoint_path)
            print(f"  --> Saved new best model to '{checkpoint_path}' (Participant F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  --> No improvement. Patience: {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
        # Fallback: Save if no dev set and train F1 improves
        if len(dev_loader) == 0 and train_f1 >= best_f1:
            best_f1 = train_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, checkpoint_path)
            print(f"  --> Saved model to '{checkpoint_path}' (Train F1={best_f1:.4f})")

    # 7. Evaluate on Test Data — PARTICIPANT-LEVEL aggregation
    print("\n--- Evaluating Best Model on Test Data ---")
    test_loader = get_test_dataloader(data_dir, batch_size=batch_size, num_workers=0)
    
    if len(test_loader.dataset) == 0:
        print("Test dataset is empty. Skipping test evaluation.")
    else:
        # Load the best model weights
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded best weights from '{checkpoint_path}' for testing.")
        
        model.eval()
        test_loss = 0.0
        participant_logits = {}
        participant_labels = {}
        
        with torch.no_grad():
            for batch_idx, (x_batch, emo_batch, y_batch) in enumerate(tqdm(test_loader, desc="[Test Evaluate]")):
                x_batch   = x_batch.to(device)
                emo_batch = emo_batch.to(device)
                y_batch   = y_batch.to(device)
                logits = model(x_batch, emo_batch)
                loss = criterion(logits, y_batch)
                test_loss += loss.item()
                
                batch_size_actual = x_batch.shape[0]
                start_idx = batch_idx * test_loader.batch_size
                logits_np = logits.cpu().numpy()
                labels_np = y_batch.cpu().numpy()
                
                for i in range(batch_size_actual):
                    sample_idx = start_idx + i
                    if sample_idx >= len(test_loader.dataset.samples):
                        break
                    file_path = test_loader.dataset.samples[sample_idx]['file']
                    if file_path not in participant_logits:
                        participant_logits[file_path] = []
                        participant_labels[file_path] = labels_np[i]
                    participant_logits[file_path].append(logits_np[i])
        
        test_preds, test_targets = [], []
        for fp in participant_logits:
            avg_logit = np.mean(participant_logits[fp])
            pred = 1.0 if avg_logit > 0.0 else 0.0
            test_preds.append(pred)
            test_targets.append(participant_labels[fp])
                
        test_loss /= len(test_loader)
        test_acc = accuracy_score(test_targets, test_preds)
        test_f1 = f1_score(test_targets, test_preds, zero_division=0)
        
        print(f"\n=== Test Results (Participant-Level, N={len(test_preds)}) ===")
        print(f"Test Loss (chunk-avg): {test_loss:.4f}")
        print(f"Test Accuracy:         {test_acc:.4f}")
        print(f"Test F1 Score:         {test_f1:.4f}")
        print("="*40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the DAIC-WOZ Hybrid Depression Model")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing processed chunks")
    parser.add_argument("--model", type=str, choices=["bi-lstm", "transformer"], default="bi-lstm", help="Model architecture to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=16, help="Hidden size for LSTM. Keep small (16-32) for small datasets.")
    parser.add_argument("--no-resume", action="store_true", help="Start training from scratch, ignoring any saved checkpoint")
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir, 
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        learning_rate=args.lr, 
        resume=not args.no_resume, 
        model_type=args.model,
        hidden_size=args.hidden_size
    )
