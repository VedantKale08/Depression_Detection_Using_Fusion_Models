import os
import pandas as pd
import numpy as np
import opensmile
from sentence_transformers import SentenceTransformer

class MultimodalFeatureExtractor:
    def __init__(self, bert_model_name='all-MiniLM-L6-v2'):
        print(f"Loading BERT model: {bert_model_name}...")
        self.bert_model = SentenceTransformer(bert_model_name)
        
        print("Loading OpenSMILE standard eGeMAPSv02 toolkit...")
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract_text_features(self, transcript_path):
        """ Extract 768-d (or 384-d depending on model) BERT embeddings (Mean, Std) from Participant text """
        if not os.path.exists(transcript_path):
            return None
        
        try:
            df = pd.read_csv(transcript_path, sep='\t')
        except Exception:
            try:
                df = pd.read_csv(transcript_path)
            except Exception:
                return None
                
        # Filter for only Participant utterances
        if 'speaker' in df.columns:
            df_part = df[df['speaker'].str.lower() == 'participant']
        else:
            df_part = df
            
        utterances = df_part['value'].dropna().tolist()
        
        if len(utterances) == 0:
            return None
            
        embeddings = self.bert_model.encode(utterances) # Shape: (num_utterances, hidden_dim)
        
        mean_emb = np.mean(embeddings, axis=0)
        std_emb = np.std(embeddings, axis=0)
        
        return np.concatenate([mean_emb, std_emb])

    def extract_audio_features(self, audio_path):
        """ Extract 88-d OpenSMILE eGeMAPS functional features from audio """
        if not os.path.exists(audio_path):
            return None
            
        try:
            # Output dataframe shape: (1, 88)
            smile_df = self.smile.process_file(audio_path)
            features = smile_df.iloc[0].values
            return features
        except Exception as e:
            print(f"Error extracting audio from {audio_path}: {e}")
            return None

    def extract_video_features(self, au_path):
        """ Extract statistical properties from OpenFace Action Units sequence """
        if not os.path.exists(au_path):
            return None
            
        df = pd.read_csv(au_path, sep=', ')  # OpenFace uses comma+space delimiter
        
        # Typically OpenFace AUs are named 'AU01_r' (intensity) or 'AU01_c' (presence)
        au_cols = [col for col in df.columns if col.startswith('AU')]
        if len(au_cols) == 0:
            return None
            
        au_data = df[au_cols].values
        
        mean_au = np.mean(au_data, axis=0)
        std_au = np.std(au_data, axis=0)
        max_au = np.max(au_data, axis=0)
        
        return np.concatenate([mean_au, std_au, max_au])

    def extract_covarep_features(self, covarep_path):
        """ Extract statistical aggregates (mean, std, max) from COVAREP 74-dim acoustic features """
        if not os.path.exists(covarep_path):
            return None
        try:
            df = pd.read_csv(covarep_path, header=None)
            data = df.values.astype(np.float32)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            # Take only the first 74 columns (standard COVAREP size)
            data = data[:, :74]
            mean_c = np.mean(data, axis=0)
            std_c  = np.std(data, axis=0)
            max_c  = np.max(data, axis=0)
            return np.concatenate([mean_c, std_c, max_c])  # 74*3 = 222 dims
        except Exception as e:
            print(f"Error extracting COVAREP from {covarep_path}: {e}")
            return None

    def load_emotion_vectors(self, text_emo_path, audio_emo_path):
        """ Load the explicitly extracted 7-dimensional emotion vectors """
        try:
            if os.path.exists(text_emo_path):
                text_emo = np.load(text_emo_path)
            else:
                return None
                
            if os.path.exists(audio_emo_path):
                audio_emo = np.load(audio_emo_path)
            else:
                return None
                
            # Flatten just in case it is saved as (1, 7)
            text_emo = text_emo.flatten()
            audio_emo = audio_emo.flatten()
            
            return np.concatenate([text_emo, audio_emo])
            
        except Exception as e:
            print(f"Error loading emotion vectors: {e}")
            return None

    def process_participant(self, participant_id, root_data_dir, emotion_vectors_dir):
        """ Process a single participant and concatenate all feature spaces """
        part_dir = os.path.join(root_data_dir, f"{participant_id}_P")
        
        transcript_path = os.path.join(part_dir, f"{participant_id}_TRANSCRIPT.csv")
        audio_path = os.path.join(part_dir, f"{participant_id}_AUDIO.wav")
        au_path = os.path.join(part_dir, f"{participant_id}_CLNF_AUs.txt")
        covarep_path = os.path.join(part_dir, f"{participant_id}_COVAREP.csv")
        
        # Depending on where the emotions are stored
        # Example: data/raw/DAIC_WOZ/300_P/300_text_emotion.npy or inside data/input/...
        text_emo_path = os.path.join(emotion_vectors_dir, f"{participant_id}_P", f"{participant_id}_text_emotion.npy")
        audio_emo_path = os.path.join(emotion_vectors_dir, f"{participant_id}_P", f"{participant_id}_audio_emotion.npy")

        # 1. Text
        text_feats = self.extract_text_features(transcript_path)
        # 2. Audio (OpenSMILE eGeMAPS)
        audio_feats = self.extract_audio_features(audio_path)
        # 3. Video (Face AUs)
        video_feats = self.extract_video_features(au_path)
        # 4. COVAREP acoustic features
        covarep_feats = self.extract_covarep_features(covarep_path)
        # 5. Emotions
        emotion_feats = self.load_emotion_vectors(text_emo_path, audio_emo_path)

        if text_feats is None or audio_feats is None or emotion_feats is None:
            print(f"Missing critical modalities for participant {participant_id}. Skipping.")
            return None
        
        # Build feature list — use zeros if face/covarep missing (non-critical)
        parts = [text_feats, audio_feats]
        parts.append(video_feats if video_feats is not None else np.zeros(60))
        parts.append(covarep_feats if covarep_feats is not None else np.zeros(222))
        parts.append(emotion_feats)
        
        # Concatenate everything into a robust 1D vector
        # Text(768) + Audio(88) + AUs(60) + COVAREP(222) + Emotions(14) = 1152
        final_vector = np.concatenate(parts)
        return final_vector
