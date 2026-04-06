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
from Text import get_text_emotion_vector

def extract_face_features(video_path, out_dir, participant_id):
    """
    Runs OpenFace Docker container to extract AUs, landmarks, pose, gaze, and HOG.
    Creates: [ID]_CLNF_features.txt, [ID]_CLNF_features3D.txt, [ID]_CLNF_AUs.txt, [ID]_CLNF_pose.txt, [ID]_CLNF_gaze.txt, [ID]_CLNF_hog.bin
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
        f"build/bin/FeatureExtraction -f /video_dir/{os.path.basename(video_path)} -out_dir /out_dir -2Dfp -3Dfp -aus -pose -gaze -hogalign"
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
                
            # 4. Gaze (gaze.txt)
            gaze_cols_expected = ['frame', 'timestamp', 'confidence', 'success', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'x_h0', 'y_h0', 'z_h0', 'x_h1', 'y_h1', 'z_h1']
            if 'gaze_0_x' in df.columns:
                df_gaze = pd.DataFrame()
                for c in ['frame', 'timestamp', 'confidence', 'success']:
                    df_gaze[c] = df[c] if c in df.columns else 0.0
                
                df_gaze['x_0'] = df['gaze_0_x']
                df_gaze['y_0'] = df['gaze_0_y']
                df_gaze['z_0'] = df['gaze_0_z']
                df_gaze['x_1'] = df['gaze_1_x']
                df_gaze['y_1'] = df['gaze_1_y']
                df_gaze['z_1'] = df['gaze_1_z']
                
                # Fill head relative gaze with 0.0 since newer OpenFace does not provide these direct coordinates
                for col in ['x_h0', 'y_h0', 'z_h0', 'x_h1', 'y_h1', 'z_h1']:
                    df_gaze[col] = 0.0
                    
                df_gaze = df_gaze[gaze_cols_expected]
                df_gaze.to_csv(os.path.join(out_dir, f"{participant_id}_CLNF_gaze.txt"), index=False)
                
            # 5. Pose (pose.txt)
            pose_cols_expected = ['frame', 'timestamp', 'confidence', 'success', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
            if 'pose_Tx' in df.columns:
                df_pose = pd.DataFrame()
                for c in ['frame', 'timestamp', 'confidence', 'success']:
                    if c in df.columns:
                        df_pose[c] = df[c]
                
                df_pose['Tx'] = df['pose_Tx']
                df_pose['Ty'] = df['pose_Ty']
                df_pose['Tz'] = df['pose_Tz']
                df_pose['Rx'] = df['pose_Rx']
                df_pose['Ry'] = df['pose_Ry']
                df_pose['Rz'] = df['pose_Rz']
                
                df_pose = df_pose[pose_cols_expected]
                df_pose.to_csv(os.path.join(out_dir, f"{participant_id}_CLNF_pose.txt"), index=False)
                
        # 6. HOG (*.hog -> _CLNF_hog.bin)
        raw_hog = os.path.join(out_dir, f"{video_stem}.hog")
        if os.path.exists(raw_hog):
            import shutil
            target_hog = os.path.join(out_dir, f"{participant_id}_CLNF_hog.bin")
            shutil.move(raw_hog, target_hog)
            
        # 7. Cleanup OpenFace residual files
        if os.path.exists(raw_csv):
            os.remove(raw_csv)
            
        of_details = os.path.join(out_dir, f"{video_stem}_of_details.txt")
        if os.path.exists(of_details):
            os.remove(of_details)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running OpenFace: {e}")
        print("Make sure Docker is installed and running.")
    except FileNotFoundError:
        print("Error: 'docker' command not found.")
        print("Please install Docker Desktop for Windows and ensure it's in your PATH.")
        print("Check https://docs.docker.com/desktop/install/windows-install/ for more details.")
        print("Skipping Face Feature extraction...")

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
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Error: 'ffmpeg' command not found.")
        print("Please install FFmpeg and ensure it's in your PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {e}")
        sys.exit(1)
        
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


def extract_covarep(wav_path, out_dir, participant_id):
    """
    Extracts 74 COVAREP-style acoustic features using librosa at 100Hz.
    Columns match the original COVAREP feature order:
      F0, VUV, NAQ, QOQ, H1H2, PSP, MDQ, peakSlope, Rd, Rd_conf, creak,
      MCEP_0..24 (25), HMPDM_0..24 (25), HMPDD_0..12 (13)
    Creates [ID]_COVAREP.csv
    """
    import librosa

    print(f"[{participant_id}] Extracting COVAREP-style features (74 features) using librosa...")
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)

        # Frame settings — 10ms hop = 100Hz, matching COVAREP default
        hop_length = int(sr * 0.01)   # 160 samples
        win_length = int(sr * 0.025)  # 400 samples
        n_fft = 512

        # --- F0 and VUV via pyin ---
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr,
            hop_length=hop_length, win_length=win_length, fill_na=0.0
        )
        F0  = np.nan_to_num(f0, nan=0.0)
        VUV = voiced_flag.astype(float)
        num_frames = len(F0)

        def _resample(v, n):
            if len(v) == n:
                return v
            return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v)), v)

        def _norm(v):
            v = _resample(v, num_frames)
            vmin, vmax = v.min(), v.max()
            return (v - vmin) / (vmax - vmin + 1e-8)

        # --- Spectral features for glottal-source proxies ---
        spec_centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        spec_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        spec_flatness  = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
        zcr            = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        rms            = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # --- Glottal-source proxies (9 features) ---
        NAQ       = _norm(rms)
        QOQ       = _norm(spec_centroid)
        H1H2      = _norm(spec_flatness)
        PSP       = _norm(spec_rolloff)
        MDQ       = _norm(spec_bandwidth)
        peakSlope = _norm(np.gradient(_resample(rms, num_frames)))
        Rd        = _resample(F0, num_frames) / (500.0 + 1e-8)
        Rd_conf   = VUV.copy()
        creak     = (_resample(zcr, num_frames) > 0.1).astype(float)

        # --- MCEP: 25 mel-cepstral coefficients (MFCC 0..24) ---
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=25,
            n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        MCEP = np.array([_resample(mfcc[i], num_frames) for i in range(25)]).T

        # --- HMPDM: harmonic phase distortion mean (25) ---
        # Approximated: chroma(12) + tonnetz(6) + harmonic-MFCC(7)
        chroma    = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        tonnetz   = librosa.feature.tonnetz(y=y, sr=sr, hop_length=hop_length)
        harm_mfcc = librosa.feature.mfcc(
            y=librosa.effects.harmonic(y), sr=sr, n_mfcc=7,
            n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        HMPDM_src = np.vstack([chroma, tonnetz, harm_mfcc])  # (25, T)
        HMPDM = np.array([_resample(HMPDM_src[i], num_frames) for i in range(25)]).T

        # --- HMPDD: harmonic phase distortion deviation (13) ---
        # Approximated: percussive-MFCC(6) + delta-MFCC(7)
        perc_mfcc  = librosa.feature.mfcc(
            y=librosa.effects.percussive(y), sr=sr, n_mfcc=6,
            n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        delta_mfcc = librosa.feature.delta(mfcc[:7, :])
        HMPDD_src  = np.vstack([perc_mfcc, delta_mfcc])  # (13, T)
        HMPDD = np.array([_resample(HMPDD_src[i], num_frames) for i in range(13)]).T

        # --- Assemble all 74 columns and save ---
        features = np.column_stack([
            F0, VUV,
            NAQ, QOQ, H1H2, PSP, MDQ, peakSlope, Rd, Rd_conf, creak,
            MCEP, HMPDM, HMPDD
        ])  # (num_frames, 74)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        pd.DataFrame(features).to_csv(
            os.path.join(out_dir, f"{participant_id}_COVAREP.csv"),
            index=False, header=False, float_format='%.6f'
        )
        print(f"[{participant_id}] COVAREP saved: {features.shape[0]} frames x {features.shape[1]} features")

    except Exception as e:
        print(f"[{participant_id}] Librosa COVAREP extraction failed: {e}. Falling back to zero array.")
        snd = parselmouth.Sound(wav_path)
        num_frames = int(snd.get_total_duration() * 100)
        pd.DataFrame(np.zeros((num_frames, 74))).to_csv(
            os.path.join(out_dir, f"{participant_id}_COVAREP.csv"), index=False, header=False
        )


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
    
    # 2.5 Enhance Audio for Inference
    try:
        from enhance_audio import enhance_audio
        enhanced_wav_path = os.path.join(out_dir, f"{participant_id}_AUDIO_ENHANCED.wav")
        enhance_audio(wav_path, enhanced_wav_path)
        wav_path = enhanced_wav_path
    except Exception as e:
        print(f"[{participant_id}] Audio enhancement failed or not installed, continuing with raw audio: {e}")
    
    # 3. Formants
    extract_formants(wav_path, out_dir, participant_id)
    
    # 4. Transcript
    extract_transcript(wav_path, out_dir, participant_id)
    
    # 5. COVAREP
    extract_covarep(wav_path, out_dir, participant_id)

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
