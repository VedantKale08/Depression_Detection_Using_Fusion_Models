import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from data_loader import get_dataloaders, get_test_dataloader

# Import the participant-level feature extraction we wrote for XGBoost
from train_xgboost import extract_participant_features

def main():
    data_dir = "data/processed"
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Ensure you are running from the project root.")
        return

    print("Loading data...")
    train_loader, dev_loader = get_dataloaders(data_dir, batch_size=32, num_workers=0)
    test_loader = get_test_dataloader(data_dir, batch_size=32, num_workers=0)

    # Extract PARTICIPANT-LEVEL features
    print("Extracting participant-level features for Logistic Regression...")
    X_train, y_train = extract_participant_features(train_loader)
    X_dev,   y_dev   = extract_participant_features(dev_loader)
    X_test,  y_test  = extract_participant_features(test_loader)

    if len(X_train) == 0:
        print("No training data found.")
        return

    print(f"Participants -> Train: {len(y_train)}, Dev: {len(y_dev)}, Test: {len(y_test)}")
    print(f"Feature vector size: {X_train.shape[1]}")

    # Standard Scaling is critical for linear models like Logistic Regression and SVM
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining Logistic Regression...")
    # class_weight='balanced' automatically handles the depressed vs non-depressed imbalance
    # C=0.1 adds L2 regularization to prevent overfitting on this small dataset
    model = LogisticRegression(class_weight='balanced', C=0.1, max_iter=1000, random_state=42)
    
    model.fit(X_train_scaled, y_train)

    print("\n--- Evaluating Logistic Regression ---")

    def evaluate(model, X, y, split_name):
        if len(X) == 0:
            return
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, zero_division=0)
        print(f"[{split_name:5s}] N={len(y):3d} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

    evaluate(model, X_train_scaled, y_train, "Train")
    if len(X_dev_scaled) > 0:
        evaluate(model, X_dev_scaled, y_dev, "Dev")
    if len(X_test_scaled) > 0:
        evaluate(model, X_test_scaled, y_test, "Test")

if __name__ == "__main__":
    main()
