import os
import glob
import numpy as np

# Suppress tensorflow warnings if any
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Audio.audio_inference import get_audio_emotion_vector_v1
from Text.text_inference import get_text_emotion_vector

def main():
    base_dir = "./test_data/DAIC_WOZ"
    
    # Find all participant folders (e.g. 300_P, 301_P)
    participant_folders = glob.glob(os.path.join(base_dir, "*_P"))
    
    if not participant_folders:
        print(f"No participant folders found in {base_dir}")
        return

    print(f"Found {len(participant_folders)} participant folders. Starting extraction...")
    
    for folder in participant_folders:
        folder_name = os.path.basename(folder)
        participant_id = folder_name.split('_')[0]
        
        print(f"\\nProcessing participant {participant_id}...")
        
        # 1. Process Audio
        audio_file = os.path.join(folder, f"{participant_id}_AUDIO.wav")
        audio_out_file = os.path.join(folder, f"{participant_id}_audio_emotion.npy")
        
        if os.path.exists(audio_file):
            print(f"  Extracting audio emotion for {participant_id}...")
            # returns a 7-element numpy array
            audio_vec = get_audio_emotion_vector_v1(audio_file)
            np.save(audio_out_file, audio_vec)
            print(f"  Saved audio vector to {audio_out_file}")
            print(f"  Vector: {audio_vec}")
        else:
            print(f"  Audio file not found: {audio_file}")
            
        # 2. Process Text
        transcript_file = os.path.join(folder, f"{participant_id}_TRANSCRIPT.csv")
        text_out_file = os.path.join(folder, f"{participant_id}_text_emotion.npy")
        
        if os.path.exists(transcript_file):
            print(f"  Extracting text emotion for {participant_id}...")
            # returns a 7-element numpy array
            text_vec = get_text_emotion_vector(transcript_file)
            np.save(text_out_file, text_vec)
            print(f"  Saved text vector to {text_out_file}")
            print(f"  Vector: {text_vec}")
        else:
            print(f"  Transcript file not found: {transcript_file}")

    print("\\nExtraction complete!")

if __name__ == "__main__":
    main()
