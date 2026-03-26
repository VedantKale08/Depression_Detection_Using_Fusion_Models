import os
import sys
import subprocess
import glob
import pandas as pd
import numpy as np
import parselmouth
import whisper
from pathlib import Path

# Add project root to python path to resolve Audio and Text imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Audio.audio_inference import get_audio_emotion_vector_v1
from Text.text_inference import get_text_emotion_vector

def extract_face_features(video_path, out_dir, participant_id):
    """
    Runs OpenFace Docker container to extract AUs and landmarks.
    Creates: [ID]_CLNF_features.txt, [ID]_CLNF_features3D.txt, [ID]_CLNF_AUs.txt
    """
    print(f"[{participant_id}] Extracting Face features using OpenFace (Docker)...")
    video_path_abs = os.path.abspath(video_path)
    out_dir_abs = os.path.abspath(out_dir)
    
    # We use docker to run OpenFace
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{os.path.dirname(video_path_abs)}:/video_dir",
        "-v", f"{out_dir_abs}:/out_dir",
        "algebr/openface:latest",
        "-c",
        f"build/bin/FeatureExtraction -f /video_dir/{os.path.basename(video_path)} -out_dir /out_dir -2Dfp -3Dfp -aus"
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # OpenFace creates files named after the video, e.g., video.csv
        # We need to rename them to match DAIC-WOZ format
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        
        # Process the generated CSV to separate features, AUs, and 3D landmarks
        raw_csv = os.path.join(out_dir, f"{video_stem}.csv")
        if os.path.exists(raw_csv):
            df = pd.read_csv(raw_csv)
            # OpenFace columns often have leading spaces
            df.columns = df.columns.str.strip()
            
            # 1. 2D features (features.txt)
            cols_2d_raw = ['frame', 'timestamp', 'confidence', 'success'] + [f'x_{i}' for i in range(68)] + [f'y_{i}' for i in range(68)]
            if all(c in df.columns for c in cols_2d_raw):
                df_2d = df[cols_2d_raw].copy()
                df_2d.columns = ['frame', 'timestamp', 'confidence', 'success'] + [f'x{i}' for i in range(68)] + [f'y{i}' for i in range(68)]
                df_2d.to_csv(os.path.join(out_dir, f"{participant_id}_CLNF_features.txt"), index=False)
                
            # 2. 3D features (features3D.txt)
            cols_3d_raw = ['frame', 'timestamp', 'confidence', 'success'] + [f'X_{i}' for i in range(68)] + [f'Y_{i}' for i in range(68)] + [f'Z_{i}' for i in range(68)]
            if all(c in df.columns for c in cols_3d_raw):
                df_3d = df[cols_3d_raw].copy()
                df_3d.columns = ['frame', 'timestamp', 'confidence', 'success'] + [f'X{i}' for i in range(68)] + [f'Y{i}' for i in range(68)] + [f'Z{i}' for i in range(68)]
                df_3d.to_csv(os.path.join(out_dir, f"{participant_id}_CLNF_features3D.txt"), index=False)
                
            # 3. AUs (AUs.txt)
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
                
            # Remove the raw CSV to clean up
            os.remove(raw_csv)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running OpenFace: {e}")
        print("Make sure Docker is installed and running.")

def extract_audio(video_path, out_dir, participant_id):
    """
    Extracts 16kHz mono WAV from video.
    """
    print(f"[{participant_id}] Extracting Audio to WAV...")
    wav_path = os.path.join(out_dir, f"{participant_id}_AUDIO.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000",
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def extract_formants(wav_path, out_dir, participant_id):
    """
    Extracts 5 formants using Parselmouth (Praat) at 100Hz.
    Creates [ID]_FORMANT.csv
    """
    print(f"[{participant_id}] Extracting Formants...")
    snd = parselmouth.Sound(wav_path)
    # Time step 0.01 for 100Hz
    formants = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5)
    
    times = formants.ts()
    formant_data = []
    for t in times:
        f1 = formants.get_value_at_time(1, t)
        f2 = formants.get_value_at_time(2, t)
        f3 = formants.get_value_at_time(3, t)
        f4 = formants.get_value_at_time(4, t)
        f5 = formants.get_value_at_time(5, t)
        formant_data.append([
            f1 if not np.isnan(f1) else 0.0,
            f2 if not np.isnan(f2) else 0.0,
            f3 if not np.isnan(f3) else 0.0,
            f4 if not np.isnan(f4) else 0.0,
            f5 if not np.isnan(f5) else 0.0
        ])
        
    df = pd.DataFrame(formant_data).round(3)
    # Using %g or %.3f to match dataset reasonable precision
    df.to_csv(os.path.join(out_dir, f"{participant_id}_FORMANT.csv"), index=False, header=False, float_format='%.3f')


def extract_transcript(wav_path, out_dir, participant_id):
    """
    Extracts transcript using Whisper.
    Creates [ID]_TRANSCRIPT.csv in tab-separated format.
    """
    print(f"[{participant_id}] Extracting Transcript using Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(wav_path, fp16=False)
    
    transcript_rows = []
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()
        transcript_rows.append({
            "start_time": f"{start:.3f}",
            "stop_time": f"{end:.3f}",
            "speaker": "Participant", # Assume single speaker for user video
            "value": text
        })
        
    df = pd.DataFrame(transcript_rows)
    out_file = os.path.join(out_dir, f"{participant_id}_TRANSCRIPT.csv")
    df.to_csv(out_file, sep='\t', index=False)


def process_video(video_path, participant_id="USER"):
    """
    Main pipeline to process a video and generate all raw features.
    """
    base_dir = "data/raw/DAIC_WOZ"
    input_base_dir = "data/input"
    out_dir = os.path.join(input_base_dir, f"{participant_id}_P")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"=== Starting Raw Feature Extraction for {participant_id} ===")
    
    # 1. OpenFace (Video)
    extract_face_features(video_path, out_dir, participant_id)
    
    # 2. Extract Audio WAV
    wav_path = extract_audio(video_path, out_dir, participant_id)
    
    # 3. Formants
    extract_formants(wav_path, out_dir, participant_id)
    
    # 4. Transcript
    extract_transcript(wav_path, out_dir, participant_id)
    
    # 5. COVAREP (Mock 74-dim for now unless Octave script is run)
    print(f"[{participant_id}] Creating dummy COVAREP output (74 features)...")
    # For a perfect match, an Octave script calling COVAREP would be used.
    # Here we create a dummy with 74 columns at 100Hz to prevent crashes.
    snd = parselmouth.Sound(wav_path)
    num_frames = int(snd.get_total_duration() * 100)
    covarep_dummy = np.zeros((num_frames, 74))
    pd.DataFrame(covarep_dummy).to_csv(os.path.join(out_dir, f"{participant_id}_COVAREP.csv"), index=False, header=False)

    # 6. Emotion Vectors
    print(f"[{participant_id}] Extracting Audio Emotion Vector...")
    audio_vec = get_audio_emotion_vector_v1(wav_path)
    np.save(os.path.join(out_dir, f"{participant_id}_audio_emotion.npy"), audio_vec)

    transcript_file = os.path.join(out_dir, f"{participant_id}_TRANSCRIPT.csv")
    print(f"[{participant_id}] Extracting Text Emotion Vector...")
    text_vec = get_text_emotion_vector(transcript_file)
    np.save(os.path.join(out_dir, f"{participant_id}_text_emotion.npy"), text_vec)

    print(f"=== Completed Extraction for {participant_id}. Outputs saved to {out_dir} ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract DAIC-WOZ styled raw features from video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (.mp4)")
    parser.add_argument("--id", type=str, default="USER", help="Participant ID to use (e.g. 500)")
    args = parser.parse_args()
    
    process_video(args.video, args.id)
