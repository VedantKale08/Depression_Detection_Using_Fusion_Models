import os
import glob
import pandas as pd
from extract_raw_features import extract_covarep, extract_formants

def process_csv(csv_path, out_dir, participant_id):
    """
    Reads the pre-extracted OpenFace CSV and splits it into AUs.txt, gaze.txt, pose.txt
    just like the original preprocessing pipeline expects.
    """
    print(f"[{participant_id}] Processing OpenFace CSV: {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # 1. AUs (AUs.txt)
    au_cols_expected = ['frame', 'timestamp', 'confidence', 'success', 
                        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r', 
                        'AU04_c', 'AU12_c', 'AU15_c', 'AU23_c', 'AU28_c', 'AU45_c']
    
    au_cols_present = [c for c in au_cols_expected if c in df.columns]
    if len(au_cols_present) > 4:
        df_au = df[au_cols_present].copy()
        
        # If any expected columns are missing, add them with 0s to maintain consistent format
        for col in au_cols_expected:
            if col not in df_au.columns:
                df_au[col] = 0.0
                
        # Ensure original column order
        df_au = df_au[au_cols_expected]
        df_au.to_csv(os.path.join(out_dir, f"{participant_id}_CLNF_AUs.txt"), index=False)
        print(f"[{participant_id}] -> Saved {participant_id}_CLNF_AUs.txt")
        
    # 2. Gaze (gaze.txt)
    gaze_cols_expected = ['frame', 'timestamp', 'confidence', 'success', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'x_h0', 'y_h0', 'z_h0', 'x_h1', 'y_h1', 'z_h1']
    if 'gaze_0_x' in df.columns:
        df_gaze = pd.DataFrame()
        for c in ['frame', 'timestamp', 'confidence', 'success']:
            df_gaze[c] = df[c] if c in df.columns else 0.0
            
        df_gaze['x_0'] = df['gaze_0_x'] if 'gaze_0_x' in df.columns else 0.0
        df_gaze['y_0'] = df['gaze_0_y'] if 'gaze_0_y' in df.columns else 0.0
        df_gaze['z_0'] = df['gaze_0_z'] if 'gaze_0_z' in df.columns else 0.0
        df_gaze['x_1'] = df['gaze_1_x'] if 'gaze_1_x' in df.columns else 0.0
        df_gaze['y_1'] = df['gaze_1_y'] if 'gaze_1_y' in df.columns else 0.0
        df_gaze['z_1'] = df['gaze_1_z'] if 'gaze_1_z' in df.columns else 0.0
        
        for col in ['x_h0', 'y_h0', 'z_h0', 'x_h1', 'y_h1', 'z_h1']:
            df_gaze[col] = 0.0
            
        df_gaze = df_gaze[gaze_cols_expected]
        df_gaze.to_csv(os.path.join(out_dir, f"{participant_id}_CLNF_gaze.txt"), index=False)
        print(f"[{participant_id}] -> Saved {participant_id}_CLNF_gaze.txt")

    # 3. Pose (pose.txt)
    pose_cols_expected = ['frame', 'timestamp', 'confidence', 'success', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
    if 'pose_Tx' in df.columns:
        df_pose = pd.DataFrame()
        for c in ['frame', 'timestamp', 'confidence', 'success']:
            df_pose[c] = df[c] if c in df.columns else 0.0
                
        df_pose['Tx'] = df['pose_Tx'] if 'pose_Tx' in df.columns else 0.0
        df_pose['Ty'] = df['pose_Ty'] if 'pose_Ty' in df.columns else 0.0
        df_pose['Tz'] = df['pose_Tz'] if 'pose_Tz' in df.columns else 0.0
        df_pose['Rx'] = df['pose_Rx'] if 'pose_Rx' in df.columns else 0.0
        df_pose['Ry'] = df['pose_Ry'] if 'pose_Ry' in df.columns else 0.0
        df_pose['Rz'] = df['pose_Rz'] if 'pose_Rz' in df.columns else 0.0
        
        df_pose = df_pose[pose_cols_expected]
        df_pose.to_csv(os.path.join(out_dir, f"{participant_id}_CLNF_pose.txt"), index=False)
        print(f"[{participant_id}] -> Saved {participant_id}_CLNF_pose.txt")

def main():
    raw_dir = "data/raw/DAIC_WOZ"
    if not os.path.exists(raw_dir):
        print(f"Directory {raw_dir} does not exist. Ensure you are running from project root.")
        return
        
    participant_folders = sorted(glob.glob(os.path.join(raw_dir, "*_P")))
    print(f"Found {len(participant_folders)} participant folders in {raw_dir}")
    
    for p_dir in participant_folders:
        participant_id = os.path.basename(p_dir).split("_")[0]
        print(f"\n--- Checking participant {participant_id} ({os.path.basename(p_dir)}) ---")
        
        # 1. Process Audio
        wav_path = os.path.join(p_dir, f"{participant_id}_AUDIO.wav")
        if os.path.exists(wav_path):
            covarep_path = os.path.join(p_dir, f"{participant_id}_COVAREP.csv")
            formant_path = os.path.join(p_dir, f"{participant_id}_FORMANT.csv")
            
            # # Formants
            # if not os.path.exists(formant_path):
            #     extract_formants(wav_path, p_dir, participant_id)
            # else:
            #     print(f"[{participant_id}] FORMANT already exists.")
                
            # COVAREP
            if not os.path.exists(covarep_path):
                extract_covarep(wav_path, p_dir, participant_id)
            else:
                print(f"[{participant_id}] COVAREP already exists.")
        else:
            print(f"[{participant_id}] Audio file not found: {wav_path}")
            
        # 2. Process OpenFace CSV inside the "features" folder (if it exists)
        features_dir = os.path.join(p_dir, "features")
        if os.path.exists(features_dir):
            # Check for slightly varied file naming conventions in the extended dataset
            pose_csv_files = glob.glob(os.path.join(features_dir, "*Pose_gaze_AUs.csv")) + \
                             glob.glob(os.path.join(features_dir, "*Pose_Gaze_AUs.csv"))
            if pose_csv_files:
                pose_csv = pose_csv_files[0]
                au_out = os.path.join(p_dir, f"{participant_id}_CLNF_AUs.txt")
                
                # If AU file isn't there, we assume pose/gaze aren't there and we extract them all
                if not os.path.exists(au_out):
                    process_csv(pose_csv, p_dir, participant_id)
                else:
                    print(f"[{participant_id}] AUs/Pose/Gaze already extracted in raw directory.")
            else:
                print(f"[{participant_id}] No *Pose_gaze_AUs.csv found in features directory.")
        else:
            print(f"[{participant_id}] No 'features' directory found.")

if __name__ == "__main__":
    main()
