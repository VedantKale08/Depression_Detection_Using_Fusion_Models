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

        # return zero vector of correct size (7 emotions)
        return np.zeros(len(output_emotions))


# quick smoke test when running the module directly
if __name__ == '__main__':
    print(get_audio_emotion_vector("./test_data/sad-ajp-1.wav"))