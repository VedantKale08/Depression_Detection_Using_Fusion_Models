import os
import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from data_loader import get_dataloaders, get_test_dataloader

def extract_participant_features(loader):
    """
    PARTICIPANT-LEVEL feature extraction.
    
    Instead of treating each chunk as an independent sample (which causes
    train/dev leakage because hundreds of chunks share the same participant label),
    we aggregate all chunks from the same participant into ONE feature vector.
    
    Strategy: compute mean, std, max, min over time axis PER CHUNK,
    then average those statistics across all chunks of the participant.
    This gives a single (210*4 + 14,) = 854-dim vector per person.
    """
    # participant_file_path -> {'feats': [...chunk_feats...], 'label': float, 'emotion': np.array}
    participant_data = {}

    for batch_idx, batch in enumerate(loader):
        if len(batch) == 3:
            x_batch, emo_batch, y_batch = batch
        else:
            x_batch, y_batch = batch
            emo_batch = torch.zeros((x_batch.shape[0], 14))

        x_np = x_batch.numpy()       # (batch, time, 210)
        emo_np = emo_batch.numpy()   # (batch, 14)
        y_np = y_batch.numpy()       # (batch,)

        # Per-chunk temporal statistics
        mean_feat = np.mean(x_np, axis=1)   # (batch, 210)
        std_feat  = np.std(x_np, axis=1)    # (batch, 210)
        max_feat  = np.max(x_np, axis=1)    # (batch, 210)
        min_feat  = np.min(x_np, axis=1)    # (batch, 210)
        chunk_feats = np.concatenate([mean_feat, std_feat, max_feat, min_feat], axis=1)  # (batch, 840)

        batch_size_actual = x_np.shape[0]
        start_idx = batch_idx * loader.batch_size

        for i in range(batch_size_actual):
            sample_idx = start_idx + i
            if sample_idx >= len(loader.dataset.samples):
                break
            file_path = loader.dataset.samples[sample_idx]['file']

            if file_path not in participant_data:
                participant_data[file_path] = {
                    'chunk_feats': [],
                    'label': y_np[i],
                    'emotion': emo_np[i]    # same for all chunks of this participant
                }
            participant_data[file_path]['chunk_feats'].append(chunk_feats[i])

    # Aggregate across chunks for each participant
    X_list, y_list = [], []
    for fp, pd in participant_data.items():
        # Average temporal stats across all chunks
        avg_chunk = np.mean(pd['chunk_feats'], axis=0)   # (840,)
        # Append session-level emotion vector
        feat = np.concatenate([avg_chunk, pd['emotion']]) # (854,)
        X_list.append(feat)
        y_list.append(pd['label'])

    if not X_list:
        return np.array([]), np.array([])

    return np.array(X_list), np.array(y_list)


def main():
    data_dir = "data/processed"
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Ensure you are running from the project root.")
        return

    print("Loading data...")
    train_loader, dev_loader = get_dataloaders(data_dir, batch_size=32, num_workers=0)
    test_loader = get_test_dataloader(data_dir, batch_size=32, num_workers=0)

    # Extract PARTICIPANT-LEVEL features
    print("Extracting participant-level features for XGBoost...")
    X_train, y_train = extract_participant_features(train_loader)
    X_dev,   y_dev   = extract_participant_features(dev_loader)
    X_test,  y_test  = extract_participant_features(test_loader)

    print(f"Participants -> Train: {len(y_train)}, Dev: {len(y_dev)}, Test: {len(y_test)}")
    print(f"Feature vector size: {X_train.shape[1]}")

    if len(X_train) == 0:
        print("No training data.")
        return

    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    scale_pos_weight = float(neg_count) / pos_count if pos_count > 0 else 1.0
    print(f"Class balance — Positive: {int(pos_count)}, Negative: {int(neg_count)}, scale_pos_weight: {scale_pos_weight:.2f}")

    print("\nTraining XGBoost (participant-level)...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,                # Shallow trees to avoid overfitting (small dataset)
        min_child_weight=5,         # Require at least 5 samples in a leaf
        gamma=1.0,                  # Minimum loss reduction to split
        subsample=0.8,              # Row subsampling
        colsample_bytree=0.7,       # Feature subsampling per tree
        reg_alpha=0.5,              # L1 regularization
        reg_lambda=2.0,             # L2 regularization
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        early_stopping_rounds=30,   # Stop if dev logloss doesn't improve for 30 rounds
        random_state=42,
    )

    eval_set = [(X_dev, y_dev)] if len(X_dev) > 0 else None
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=50
    )

    print("\n--- Evaluating Models (Participant-Level) ---")

    def evaluate(model, X, y, split_name):
        if len(X) == 0:
            return
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, zero_division=0)
        print(f"[{split_name:5s}] N={len(y):3d} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

    evaluate(model, X_train, y_train, "Train")
    evaluate(model, X_dev,   y_dev,   "Dev")
    evaluate(model, X_test,  y_test,  "Test")


if __name__ == "__main__":
    main()
