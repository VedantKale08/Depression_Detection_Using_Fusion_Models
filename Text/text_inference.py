import numpy as np
import os

emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def get_text_emotion_vector(transcript_file):
    """
    Dummy text emotion model that returns a random distribution over 7 emotions.
    To be replaced with actual BERT model later.
    """
    if not os.path.exists(transcript_file):
        print(f"Transcript file {transcript_file} not found.")
        return np.zeros(len(emotion_labels))

    # Generate random probabilities that sum to 1
    random_probs = np.random.rand(len(emotion_labels))
    random_probs /= random_probs.sum()
    
    return random_probs
