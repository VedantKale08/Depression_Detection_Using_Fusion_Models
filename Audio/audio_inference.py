import numpy as np
import librosa
from keras.models import load_model

# ---------------- LOAD MODEL ----------------
audio_model = load_model("./working/audio_emotion_model.h5")

# Model was trained with these 9 emotion labels in this order
emotion_labels = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise", "surprised"]

# Only use these 7 in the final output
output_emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ---------------- FEATURE EXTRACTION ----------------
def extract_features(data, sr=22050, max_pad_len=216):

    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc


# ---------------- MAIN VECTOR FUNCTION ----------------
def get_audio_emotion_vector(file_path):

    try:
        data, sr = librosa.load(file_path, duration=3, offset=0.5)
        feat = extract_features(data,sr)
        feat = np.expand_dims(feat,axis=(0,-1))

        probs = audio_model.predict(feat, verbose=0)[0]

        # map 9 probabilities to their labels then select the 7 we care about
        emotion_dict = {emotion: float(prob) for emotion, prob in zip(emotion_labels, probs)}
        final_vector = np.array([emotion_dict[emotion] for emotion in output_emotions])
        return final_vector

    except Exception as e:
        print(f"Audio inference error: {e}")

        return np.zeros(len(output_emotions))


# ---------------- LONG AUDIO VECTOR FUNCTION ----------------
def get_audio_emotion_vector_v1(file_path, chunk_duration=3, overlap=0, batch_size=32):
    try:
        # Loading audio file
        data, sr = librosa.load(file_path, sr=22050)
        
        chunk_samples = int(chunk_duration * sr)
        step_samples = int((chunk_duration - overlap) * sr)
        
        # Creating chunks
        if len(data) < chunk_samples:
            chunks = [data]
        else:
            chunks = [data[i:i + chunk_samples] for i in range(0, len(data) - chunk_samples + 1, step_samples)]
            
        if not chunks:
            return np.zeros(len(output_emotions))

        features = []
        for chunk in chunks:
            feat = extract_features(chunk, sr)
            features.append(feat)
            
        features = np.array(features)
        features = np.expand_dims(features, axis=-1)
        
        # Predict all chunks in batches
        all_probs = []
        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size]
            batch_probs = audio_model.predict(batch, verbose=0)
            all_probs.extend(batch_probs)
            
        all_probs = np.array(all_probs)
        
        avg_probs = np.mean(all_probs, axis=0)

        emotion_dict = {emotion: float(prob) for emotion, prob in zip(emotion_labels, avg_probs)}
        final_vector = np.array([emotion_dict[emotion] for emotion in output_emotions])
        return final_vector

    except Exception as e:
        print(f"Audio video inference error: {e}")

        return np.zeros(len(output_emotions))


if __name__ == '__main__':
    print(get_audio_emotion_vector_v1("./test_data/sad-ajp-1.wav"))