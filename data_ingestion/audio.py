# data_ingestion/audio.py
import librosa
import numpy as np

def load_audio_features(audio_file, sample_rate=22050):
    """Extract MFCC features from an audio file."""
    signal, sr = librosa.load(audio_file, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1)

if __name__ == "__main__":
    features = load_audio_features("data/audio.wav")
    print("Extracted MFCC features:", features)
