EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)

LABEL_TO_ID = {label: i for i, label in enumerate(EMOTIONS)}
ID_TO_LABEL = {i: label for i, label in enumerate(EMOTIONS)}

# Model settings

MODEL_NAME = "roberta-base" # You can also use "distilbert-base-uncased" for a smaller model
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 6
LEARNING_RATE = 2e-5

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "working", "saved_model")

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
