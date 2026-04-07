import numpy as np
import os
import pandas as pd
from Text.text_inference import TextEmotionPredictor

emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
_predictor = None

def get_text_emotion_vector(transcript_file):
    """
    Generates text emotion vector for a participant's transcript.
    Filters out Bot/Interviewer text and extracts exact participant dialog.
    Averages predictions over all participant text segments to produce a single 7-class prob vector.
    
    DAIC-WOZ transcript format: comma-separated Start_Time,End_Time,Text,Confidence
    All rows are participant speech (single-mic recording).
    """
    global _predictor
    
    if not os.path.exists(transcript_file):
        print(f"Transcript file {transcript_file} not found.")
        return np.zeros(len(emotion_labels))

    try:
        # Auto-detect separator (DAIC-WOZ uses comma, some older files use tab)
        with open(transcript_file, 'r') as f:
            first_line = f.readline()
        sep = '\t' if '\t' in first_line else ','
        df = pd.read_csv(transcript_file, sep=sep)
        
        # Strip spaces from column names to safely retrieve values
        df.columns = df.columns.str.strip()
        
        # Filter for only participant text if a speaker column exists.
        # In DAIC-WOZ per-participant-mic CSVs there is NO speaker column –
        # every row is already participant speech, so we use all rows.
        if 'speaker' in df.columns:
            is_participant = df['speaker'].astype(str).str.lower().str.strip() == 'participant'
            participant_df = df[is_participant]
        else:
            participant_df = df
            
        # Column name for the text content varies by format
        text_col = None
        for candidate in ['Text', 'value', 'text', 'utterance']:
            if candidate in participant_df.columns:
                text_col = candidate
                break

        if text_col is None:
            print(f"No text column found in {transcript_file}. Columns: {list(df.columns)}")
            return np.zeros(len(emotion_labels))

        texts = participant_df[text_col].dropna().astype(str).tolist()
        # Filter out very short fragments (noise/filler)
        texts = [t.strip() for t in texts if len(t.strip()) > 2]
            
        if not texts:
            print(f"No participant text found in {transcript_file}.")
            return np.zeros(len(emotion_labels))

        # Lazy load model
        if _predictor is None:
            _predictor = TextEmotionPredictor()

        # Join all utterances into a single continuous text string
        full_text = " ".join(texts)
        words = full_text.split()
        
        # Split into chunks of roughly 100 words to provide maximum context
        # while respecting the model's 128 max token length limits.
        chunk_size = 100
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        all_probs = []
        for chunk in chunks:
            if chunk.strip():
                prob_list, _ = _predictor.predict_probabilities(chunk)
                all_probs.append(prob_list)
        
        if not all_probs:
            return np.zeros(len(emotion_labels))
            
        # Average the predicted probabilities across all chunks
        avg_probs = np.mean(all_probs, axis=0)
        
        # Ensure array sums exactly to 1.0
        if avg_probs.sum() > 0:
            avg_probs = avg_probs / avg_probs.sum()
            
        return avg_probs
        
    except Exception as e:
        print(f"Error processing text inference for {transcript_file}: {e}")
        return np.zeros(len(emotion_labels))
