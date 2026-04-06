import os
import glob
import pandas as pd
import numpy as np
import shutil
import sys
from pathlib import Path

# Add project root to python path to resolve Audio and Text imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from extract_raw_features import extract_covarep, extract_formants
from Audio.audio_inference import get_audio_emotion_vector_v1
from Text import get_text_emotion_vector

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


def remove_nested_hierarchy(p_dir, participant_id):
    """
    Often DAIC_WOZ archives extract into a nested folder, e.g., 715_P/715_P/...
    This function elevates all files to the parent p_dir and removes the inner folder.
    """
    nested_dir = os.path.join(p_dir, f"{participant_id}_P")
    if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
        print(f"[{participant_id}] Found nested folder structure. Moving files up to parent directory...")
        for item in os.listdir(nested_dir):
            src = os.path.join(nested_dir, item)
            dst = os.path.join(p_dir, item)
            try:
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, p_dir)
            except Exception as e:
                print(f"[{participant_id}] Error moving {item}: {e}")
        try:
            os.rmdir(nested_dir)
        except OSError:
            shutil.rmtree(nested_dir, ignore_errors=True)

def main():
    raw_dir = "data/raw/DAIC_WOZ"
    if not os.path.exists(raw_dir):
        print(f"Directory {raw_dir} does not exist. Ensure you are running from project root.")
        return
        
    participant_folders = sorted(glob.glob(os.path.join(raw_dir, "*_P")))
    print(f"Found {len(participant_folders)} participant folders in {raw_dir}")
    
    for p_dir in participant_folders:
        participant_id = os.path.basename(p_dir).split("_")[0]
        print(f"\\n--- Checking participant {participant_id} ({os.path.basename(p_dir)}) ---")
        
        # 0.1 Flatten nested participant folder if present
        remove_nested_hierarchy(p_dir, participant_id)
        
        # 0.2 Ensure transcript matches 300_TRANSCRIPT.csv format
        transcript_out = os.path.join(p_dir, f"{participant_id}_TRANSCRIPT.csv")
        if not os.path.exists(transcript_out):
            trans_candidates = [f"{participant_id}_Transcript.csv", f"{participant_id}_transcript.csv"]
            for cand in trans_candidates:
                cand_path = os.path.join(p_dir, cand)
                if os.path.exists(cand_path):
                    os.rename(cand_path, transcript_out)
                    print(f"[{participant_id}] Renamed {cand} to {participant_id}_TRANSCRIPT.csv")
                    break

        # 1. Process Audio & Text (+ COVAREP + Emotion Vectors)
        wav_path = os.path.join(p_dir, f"{participant_id}_AUDIO.wav")
        if os.path.exists(wav_path):
            covarep_path = os.path.join(p_dir, f"{participant_id}_COVAREP.csv")
            audio_emo_out = os.path.join(p_dir, f"{participant_id}_audio_emotion.npy")
            
            # COVAREP extraction
            if not os.path.exists(covarep_path):
                extract_covarep(wav_path, p_dir, participant_id)
            else:
                print(f"[{participant_id}] COVAREP already exists.")
                
            # Audio emotion
            if not os.path.exists(audio_emo_out):
                print(f"[{participant_id}] Extracting audio emotion vector...")
                vec = get_audio_emotion_vector_v1(wav_path)
                np.save(audio_emo_out, vec)
                print(f"[{participant_id}] -> Saved {participant_id}_audio_emotion.npy")
        else:
            print(f"[{participant_id}] Audio file not found: {wav_path}")
            
        # Text emotion
        text_emo_out = os.path.join(p_dir, f"{participant_id}_text_emotion.npy")
        if os.path.exists(transcript_out) and not os.path.exists(text_emo_out):
            print(f"[{participant_id}] Extracting text emotion vector...")
            vec = get_text_emotion_vector(transcript_out)
            np.save(text_emo_out, vec)
            print(f"[{participant_id}] -> Saved {participant_id}_text_emotion.npy")

        # 2. Process OpenFace CSV (Search dynamically in both root and 'features' folder)
        pose_csv = None
        search_dirs = [p_dir, os.path.join(p_dir, "features")]
        for sdir in search_dirs:
            if os.path.exists(sdir):
                for fname in os.listdir(sdir):
                    lname = fname.lower()
                    if ("openface" in lname or "pose_gaze" in lname) and lname.endswith(".csv"):
                        pose_csv = os.path.join(sdir, fname)
                        break
            if pose_csv:
                break
        
        au_out = os.path.join(p_dir, f"{participant_id}_CLNF_AUs.txt")
        if pose_csv:
            # If AU file isn't there, extract
            if not os.path.exists(au_out):
                try:
                    process_csv(pose_csv, p_dir, participant_id)
                except Exception as e:
                    print(f"[{participant_id}] Failed to process OpenFace CSV: {e}")
            else:
                print(f"[{participant_id}] AUs/Pose/Gaze already extracted in raw directory.")
        else:
            print(f"[{participant_id}] No OpenFace CSV found. Cannot extract AUs, gaze, pose.")
            
        # 3. Clean up unneeded nested hierarchy SAFELY
        features_dir = os.path.join(p_dir, "features")
        # Only remove features directory if the output actually exists
        if os.path.exists(features_dir) and os.path.exists(au_out):
            try:
                shutil.rmtree(features_dir)
                print(f"[{participant_id}] Removed nested 'features' hierarchy to match standard 300_P format.")
            except Exception as e:
                print(f"[{participant_id}] Could not remove 'features' directory: {e}")
            
        # Also clean up unneeded tar.gz archives (only if we successfully processed the folder!)
        if os.path.exists(wav_path):
            for tar_file in glob.glob(os.path.join(p_dir, "*.tar.gz")):
                try:
                    os.remove(tar_file)
                    print(f"[{participant_id}] Removed unused archive: {os.path.basename(tar_file)}")
                except Exception as e:
                    pass

if __name__ == "__main__":
    main()
