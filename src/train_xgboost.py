import os
import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from data_loader import get_dataloaders, get_test_dataloader

def extract_features(loader):
    X_list = []
    y_list = []
    
    for batch in loader:
        if len(batch) == 3:
            x_batch, emo_batch, y_batch = batch
        else:
            x_batch, y_batch = batch
            emo_batch = torch.zeros((x_batch.shape[0], 14))

        x_batch = x_batch.numpy() # (batch, time, 210)
        emo_batch = emo_batch.numpy() # (batch, 14)
        y_batch = y_batch.numpy() # (batch,)
        
        # calculate statistics over time axis (axis=1)
        mean_feat = np.mean(x_batch, axis=1) # (batch, 210)
        std_feat = np.std(x_batch, axis=1) # (batch, 210)
        max_feat = np.max(x_batch, axis=1) # (batch, 210)
        min_feat = np.min(x_batch, axis=1) # (batch, 210)
        
        flat_feats = np.concatenate([mean_feat, std_feat, max_feat, min_feat, emo_batch], axis=1)
        
        X_list.append(flat_feats)
        y_list.append(y_batch)
        
    if not X_list:
        return np.array([]), np.array([])
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

def main():
    data_dir = "data/processed"
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Ensure you are running from the project root.")
        return

    print("Loading data...")
    train_loader, dev_loader = get_dataloaders(data_dir, batch_size=32, num_workers=0)
    test_loader = get_test_dataloader(data_dir, batch_size=32, num_workers=0)
    
    # Extract features
    print("Extracting features for XGBoost...")
    X_train, y_train = extract_features(train_loader)
    X_dev, y_dev = extract_features(dev_loader)
    X_test, y_test = extract_features(test_loader)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    if len(X_train) == 0:
        print("No training data.")
        return
        
    print("Training XGBoost...")
    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    scale_pos_weight = float(neg_count) / pos_count if pos_count > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    print("\n--- Evaluating Models ---")
    
    def evaluate(model, X, y, split_name):
        if len(X) == 0:
            return
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, zero_division=0)
        print(f"[{split_name}] Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
    evaluate(model, X_train, y_train, "Train")
    evaluate(model, X_dev, y_dev, "Dev")
    evaluate(model, X_test, y_test, "Test")

if __name__ == "__main__":
    main()
