import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from data_loader import get_dataloaders, get_test_dataloader
from train_xgboost import extract_features
from model import DepressionHybridModel

# Set high-quality plotting for research papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,        # High resolution for papers
    'savefig.dpi': 300
})

def main():
    data_dir = "data/processed"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found.")
        return

    print("1. Loading and extracting features...")
    train_loader, dev_loader = get_dataloaders(data_dir, batch_size=32, num_workers=0)
    test_loader = get_test_dataloader(data_dir, batch_size=32, num_workers=0)

    X_train, y_train = extract_features(train_loader)
    X_test, y_test = extract_features(test_loader)

    if len(X_train) == 0 or len(X_test) == 0:
        print("Not enough data to run analytics.")
        return

    # Scale features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Class balance weights
    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    xgb_weight = float(neg_count) / pos_count if pos_count > 0 else 1.0

    print("2. Training Models...")
    # Dictionary to store trained models
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', C=0.1, max_iter=1000, random_state=42),
        'Linear SVM': SVC(kernel='linear', class_weight='balanced', C=0.05, probability=True, random_state=42),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, min_child_weight=5, 
            gamma=1.0, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, 
            reg_lambda=2.0, scale_pos_weight=xgb_weight, eval_metric='logloss', random_state=42
        )
    }

    # Store predictions
    test_probs = {}
    test_preds = {}

    for name, model in models.items():
        if name in ['Logistic Regression', 'Linear SVM']:
            model.fit(X_train_scaled, y_train)
            test_probs[name] = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_scaled) # SVM can use decision function if proba fails
            # Force Probabilities to be 0-1 for SVM decision function if predicting proba fails (or we can just ensure probability=True)
            if name == 'Linear SVM' and not hasattr(model, "predict_proba"):
                test_probs[name] = 1 / (1 + np.exp(-test_probs[name]))
            
            test_preds[name] = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            test_probs[name] = model.predict_proba(X_test)[:, 1]
            test_preds[name] = model.predict(X_test)

    print("3. Evaluating LSTM Model...")
    # Add LSTM if weights exist (using the hidden_size=16 configuration)
    lstm_weights = "weights/best_hybrid_model.pth"
    if os.path.exists(lstm_weights):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lstm_model = DepressionHybridModel(input_size=210, hidden_size=64, num_layers=1)
        checkpoint = torch.load(lstm_weights, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            lstm_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            lstm_model.load_state_dict(checkpoint)
            
        lstm_model.to(device)
        lstm_model.eval()
        
        lstm_probs_list = []
        lstm_preds_list = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if len(batch) == 3:
                    x_batch, emo_batch, y_batch = batch
                else:
                    x_batch, y_batch = batch
                    emo_batch = torch.zeros((x_batch.shape[0], 14))
                
                x_batch = x_batch.to(device)
                emo_batch = emo_batch.to(device)
                
                logits = lstm_model(x_batch, emo_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (logits > 0).float().cpu().numpy().astype(int)
                
                lstm_probs_list.extend(probs)
                lstm_preds_list.extend(preds)
                
        test_probs['Bi-LSTM'] = np.array(lstm_probs_list)
        test_preds['Bi-LSTM'] = np.array(lstm_preds_list)
        print(f"  --> Successfully added Bi-LSTM predictions (Count: {len(lstm_preds_list)} chunks).")
    else:
        print(f"  --> Bi-LSTM weights not found at {lstm_weights}. Skipping LSTM plot.")

    print("4. Generating Research Quality Plots...")

    # --- PLOT 1: ROC Curve Comparison ---
    plt.figure(figsize=(8, 6))
    for name in models.keys():
        fpr, tpr, _ = roc_curve(y_test, test_probs[name])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Comparison')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curve_comparison.png'))
    plt.close()

    # --- PLOT 2: Precision-Recall Curve ---
    plt.figure(figsize=(8, 6))
    baseline = sum(y_test == 1) / len(y_test)
    for name in models.keys():
        precision, recall, _ = precision_recall_curve(y_test, test_probs[name])
        pr_auc = average_precision_score(y_test, test_probs[name])
        plt.plot(recall, precision, lw=2, label=f'{name} (AP = {pr_auc:.3f})')
        
    plt.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pr_curve_comparison.png'))
    plt.close()

    # --- PLOT 3: Confusion Matrices ---
    # Dynamically scale grid size based on number of models
    n_models = len(models.keys()) + (1 if 'Bi-LSTM' in test_probs else 0)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
        
    for i, name in enumerate(test_probs.keys()):
        cm = confusion_matrix(y_test, test_preds[name])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Healthy', 'Depressed'],
                    yticklabels=['Healthy', 'Depressed'],
                    cbar=False, annot_kws={"size": 14})
        axes[i].set_title(f'{name}\nF1: {f1_score(y_test, test_preds[name]):.3f}', size=14)
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'))
    plt.close()

    # --- PLOT 4: Probability Distribution (KDE) ---
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
        
    for i, name in enumerate(test_probs.keys()):
        sns.kdeplot(x=test_probs[name][y_test == 0], fill=True, label='Healthy (0)', color='green', ax=axes[i])
        sns.kdeplot(x=test_probs[name][y_test == 1], fill=True, label='Depressed (1)', color='red', ax=axes[i])
        axes[i].axvline(0.5, color='black', linestyle='--', alpha=0.5) # Decision boundary
        axes[i].set_title(f'{name} Output Distribution')
        axes[i].set_xlabel('Predicted Probability of Depression')
        axes[i].set_ylabel('Density')
        if i == 0:
            axes[i].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'probability_distributions.png'))
    plt.close()

    print(f"\nDone! High-resolution research plots saved to the '{os.path.abspath(results_dir)}' directory.")
    print("Files created:")
    print("  - roc_curve_comparison.png")
    print("  - pr_curve_comparison.png")
    print("  - confusion_matrices.png")
    print("  - probability_distributions.png")

if __name__ == "__main__":
    import warnings
    # Suppress seaborn future warnings for cleaner output
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()
