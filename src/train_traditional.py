import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

def load_data(save_dir="/home/vedant/MyProjects/FInalYearProject/Audio+Face/data/processed_features"):
    """ Load the numpy arrays for model training """
    try:
        X_train = np.load(os.path.join(save_dir, "X_train_scaled.npy"))
        y_train = np.load(os.path.join(save_dir, "y_train.npy"))
        
        X_dev = np.load(os.path.join(save_dir, "X_dev_scaled.npy"))
        y_dev = np.load(os.path.join(save_dir, "y_dev.npy"))
        
        return X_train, y_train, X_dev, y_dev
    except Exception as e:
        print(f"Error loading arrays: {e}")
        return None, None, None, None

def evaluate_model(name, y_true, y_pred):
    print(f"=== {name} Performance (Dev Set) ===")
    print(classification_report(y_true, y_pred))
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred), "\n")

def run_pipeline():
    X_train, y_train, X_dev, y_dev = load_data()
    if X_train is None or len(X_train) == 0:
        print("Data is not ready yet! Please execute build_dataset.py first.")
        return
        
    print(f"Raw Input shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Raw Input shapes - X_dev: {X_dev.shape}, y_dev: {y_dev.shape}")
    
    # 1. Dimensionality Reduction (PCA)
    print("\n[Step 1] Applying PCA (Capturing 95% Variance)")
    pca = PCA(n_components=0.95, random_state=42)
    # Important: PCA fit on train only
    X_train_pca = pca.fit_transform(X_train)
    X_dev_pca = pca.transform(X_dev)
    print(f"Feature Space Reduced from {X_train.shape[1]} to {X_train_pca.shape[1]} dimensions.")

    # 2. Imbalance Handling (SMOTE)
    print(f"\n[Step 2] Applying SMOTE to balance Training Set classes")
    print(f"Class distribution before SMOTE: {np.bincount(y_train.astype(int))}")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pca, y_train)
    print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled.astype(int))}")

    # 3. Model Training & Evaluation
    print("\n[Step 3] Training ML Classifiers\n")
    
    # --- Logistic Regression ---
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_resampled, y_train_resampled)
    y_pred_lr = lr.predict(X_dev_pca)
    evaluate_model("Logistic Regression", y_dev, y_pred_lr)
    
    # --- Support Vector Machine ---
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train_resampled, y_train_resampled)
    y_pred_svm = svm.predict(X_dev_pca)
    evaluate_model("SVM (RBF Kernel)", y_dev, y_pred_svm)
    
    # --- XGBoost ---
    xgb = XGBClassifier(
        scale_pos_weight=len(y_train_resampled[y_train_resampled==0])/len(y_train_resampled[y_train_resampled==1]), 
        random_state=42,
        eval_metric='logloss'
    )
    xgb.fit(X_train_resampled, y_train_resampled)
    y_pred_xgb = xgb.predict(X_dev_pca)
    evaluate_model("XGBoost", y_dev, y_pred_xgb)

if __name__ == "__main__":
    run_pipeline()
