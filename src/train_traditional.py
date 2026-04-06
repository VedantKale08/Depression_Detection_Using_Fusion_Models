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

def _default_save_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, "data", "processed_features")

def load_data(save_dir=None):
    """ Load the numpy arrays for model training """
    if save_dir is None:
        save_dir = _default_save_dir()
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

def predict_with_threshold(model, X, threshold=0.35):
    """ Use probability threshold instead of default 0.5 to improve depression recall """
    proba = model.predict_proba(X)[:, 1]  # Probability of class 1 (depressed)
    return (proba >= threshold).astype(int)

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
    THRESHOLD = 0.35  # Lower threshold → catch more depressed cases (improves recall for class 1)
    
    # --- Logistic Regression with grid search ---
    from sklearn.model_selection import GridSearchCV
    lr_params = {'C': [0.01, 0.1, 1, 5, 10]}
    lr_cv = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
                         lr_params, scoring='f1_macro', cv=5)
    lr_cv.fit(X_train_resampled, y_train_resampled)
    lr = lr_cv.best_estimator_
    print(f"Best LR C={lr_cv.best_params_['C']}")
    y_pred_lr = predict_with_threshold(lr, X_dev_pca, THRESHOLD)
    evaluate_model("Logistic Regression", y_dev, y_pred_lr)
    
    # --- Support Vector Machine with grid search ---
    svm_params = {'C': [0.1, 1, 5, 10], 'gamma': ['scale', 'auto']}
    svm_cv = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
                          svm_params, scoring='f1_macro', cv=5)
    svm_cv.fit(X_train_resampled, y_train_resampled)
    svm = svm_cv.best_estimator_
    print(f"Best SVM C={svm_cv.best_params_['C']}, gamma={svm_cv.best_params_['gamma']}")
    y_pred_svm = predict_with_threshold(svm, X_dev_pca, THRESHOLD)
    evaluate_model("SVM (RBF Kernel)", y_dev, y_pred_svm)
    
    # --- XGBoost ---
    xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
    xgb_cv = GridSearchCV(XGBClassifier(
        scale_pos_weight=len(y_train[y_train==0])/max(1,len(y_train[y_train==1])), 
        random_state=42, eval_metric='logloss'), xgb_params, scoring='f1_macro', cv=5)
    xgb_cv.fit(X_train_resampled, y_train_resampled)
    xgb = xgb_cv.best_estimator_
    print(f"Best XGB params: {xgb_cv.best_params_}")
    y_pred_xgb = predict_with_threshold(xgb, X_dev_pca, THRESHOLD)
    evaluate_model("XGBoost", y_dev, y_pred_xgb)

if __name__ == "__main__":
    run_pipeline()
