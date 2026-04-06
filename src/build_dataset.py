import os
import glob
import pandas as pd
import numpy as np
import scipy.io as sio  # Just in case we need MAT files
from tqdm import tqdm
from feature_extractor import MultimodalFeatureExtractor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def process_split(split_df, extractor, root_data_dir, emotion_vectors_dir):
    """
    Process all participants in a split given the dataframe containing their IDs.
    """
    X = []
    y = []
    metadata = []
    
    # Check DAIC-WOZ usual columns: Participant_ID, PHQ8_Binary, PHQ8_Score
    id_col = 'Participant_ID' if 'Participant_ID' in split_df.columns else split_df.columns[0]
    binary_label_col = 'PHQ8_Binary'
    
    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
        participant_id = int(row[id_col])
        label = row[binary_label_col] if binary_label_col in split_df.columns else None
        
        # Extract everything
        feature_vector = extractor.process_participant(
            participant_id=participant_id,
            root_data_dir=root_data_dir,
            emotion_vectors_dir=emotion_vectors_dir
        )
        
        if feature_vector is not None:
            X.append(feature_vector)
            y.append(label)
            metadata.append(participant_id)
        else:
            print(f"Skipping Participant {participant_id} due to extraction failure.")

    if len(X) == 0:
        return np.array([]), np.array([]), np.array([])
        
    return np.vstack(X), np.array(y), np.array(metadata)


def build_and_save_datasets():
    # Paths configuration
    ROOT_DATA_DIR = "/home/vedant/MyProjects/FInalYearProject/Audio+Face/data/raw/DAIC_WOZ"
    EMOTION_DIR = "/home/vedant/MyProjects/FInalYearProject/Audio+Face/data/input" # Emotion npys are in DAIC_WOZ or input!
    # Update EMOTION_DIR based on where most .npy are stored. We'll search in both input/ and raw/DAIC_WOZ/ 
    EMOTION_DIR = ROOT_DATA_DIR 
    
    SAVE_DIR = "/home/vedant/MyProjects/FInalYearProject/Audio+Face/data/processed_features"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Init Extractor
    print("Initializing Extractor Models...")
    extractor = MultimodalFeatureExtractor()
    
    # 2. Iterate Splits
    splits = ['train', 'dev', 'test']
    
    dataset_dict = {}
    
    print("\nStarting extraction. This may take some time depending on hardware...")
    for split in splits:
        split_path = os.path.join(ROOT_DATA_DIR, f"{split}_split_Depression_AVEC2017.csv")
        
        if not os.path.exists(split_path):
            print(f"Split file missing: {split_path}")
            continue
            
        print(f"\nProcessing {split.upper()} split...")
        df = pd.read_csv(split_path)
        
        # In DAIC_WOZ, test split doesn't have labels provided originally (used for server eval)
        # But we extract features anyway
        X, y, ids = process_split(df, extractor, ROOT_DATA_DIR, EMOTION_DIR)
        
        dataset_dict[split] = {'X': X, 'y': y, 'ids': ids}
        print(f"Extracted dims for {split}: X={X.shape}, y={y.shape}")

    # 3. Post-processing: Imputation and Normalization
    print("\nRunning imputation and scaling based on TRAIN set statistics...")
    if 'train' in dataset_dict and dataset_dict['train']['X'].size > 0:
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        
        # Fit on Train
        X_train = dataset_dict['train']['X']
        X_train_imp = imputer.fit_transform(X_train)
        X_train_scaled = scaler.fit_transform(X_train_imp)
        
        dataset_dict['train']['X_scaled'] = X_train_scaled
        
        # Transform Dev/Test
        for s in ['dev', 'test']:
            if s in dataset_dict and dataset_dict[s]['X'].size > 0:
                X_imp = imputer.transform(dataset_dict[s]['X'])
                X_scaled = scaler.transform(X_imp)
                dataset_dict[s]['X_scaled'] = X_scaled

    # 4. Save to Disk
    for s, data in dataset_dict.items():
        if data['X'].size > 0:
            np.save(os.path.join(SAVE_DIR, f"X_{s}.npy"), data['X'])
            np.save(os.path.join(SAVE_DIR, f"X_{s}_scaled.npy"), data.get('X_scaled', data['X']))
            if data['y'].size > 0 and data['y'][0] is not None:
                np.save(os.path.join(SAVE_DIR, f"y_{s}.npy"), data['y'])
            np.save(os.path.join(SAVE_DIR, f"ids_{s}.npy"), data['ids'])
    
    print(f"\nFinished. Formatted vectors and labels saved to {SAVE_DIR}")


if __name__ == "__main__":
    build_and_save_datasets()
