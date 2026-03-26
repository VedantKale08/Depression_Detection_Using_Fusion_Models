#!/bin/bash

echo "Installing system dependencies (Octave, ffmpeg)..."
sudo apt-get update
sudo apt-get install -y octave octave-signal octave-control ffmpeg

echo "Downloading COVAREP repository (for Audio features)..."
if [ ! -d "covarep" ]; then
    git clone https://github.com/covarep/covarep.git
fi

echo "Pulling OpenFace Docker image (for Face features)..."
# Using OpenFace docker is much easier than compiling on Linux
docker pull algebr/openface:latest

echo "Installing Python dependencies (Whisper, Parselmouth)..."
pip install praat-parselmouth openai-whisper ffmpeg-python pandas librosa
