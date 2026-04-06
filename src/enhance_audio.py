import librosa
import soundfile as sf
try:
    import noisereduce as nr
except ImportError:
    print("Please install noisereduce: pip install noisereduce")
    raise

try:
    import pyloudnorm as pyln
except ImportError:
    print("Please install pyloudnorm: pip install pyloudnorm")
    raise

def enhance_audio(input_file, output_file, target_db=-1.0):
    """
    Enhances a user-provided audio file by removing background noise
    and normalizing the volume to a target peak.
    
    Args:
        input_file (str): Path to the raw, noisy input audio/video file.
        output_file (str): Path to save the cleaned audio.
        target_db (float): Target peak normalization in dB. Default is -1.0 dB.
    """
    print(f"Loading user audio: {input_file}...")
    # Load audio using librosa (automatically handles sample rate conversion if needed)
    audio_data, rate = librosa.load(input_file, sr=None)
    
    print("1. Estimating noise profile and performing spectral gating reduction...")
    # Perform automated noise reduction algorithm
    # stationary=True assumes the background noise (e.g. computer fans) is constant.
    reduced_noise = nr.reduce_noise(y=audio_data, sr=rate, stationary=True)
    
    print(f"2. Performing peak normalization to {target_db} dB...")
    # Peak normalize the audio to ensure the user's voice is loud and clear for Whisper
    normalized_audio = pyln.normalize.peak(reduced_noise, target_db)
    
    print(f"Saving enhanced audio to: {output_file}...")
    sf.write(output_file, normalized_audio, rate)
    print("Audio enhancement complete!")


