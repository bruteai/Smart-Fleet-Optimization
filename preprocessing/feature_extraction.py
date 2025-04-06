import cv2
import librosa

def add_can_features(df):
    """Add additional features from CAN data (e.g., rolling average of fuel_rate)."""
    df['fuel_rate'] = df['fuel_rate'].astype(float)
    df['avg_fuel'] = df['fuel_rate'].rolling(window=5, min_periods=1).mean()
    return df

def add_gps_features(df):
    """Calculate differences in latitude and longitude for GPS data."""
    df['lat_diff'] = df['latitude'].diff()
    df['lon_diff'] = df['longitude'].diff()
    return df

def extract_video_features(frames):
    """Derive simple features from video frames by computing the average color."""
    feats = [cv2.mean(frame)[:3] for frame in frames]
    return feats

def extract_audio_features(audio_file):
    """Extract MFCC features from audio; same as in data ingestion."""
    signal, sr = librosa.load(audio_file, sr=22050)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1)

if __name__ == "__main__":
    import pandas as pd
    sample_df = pd.DataFrame({'fuel_rate': [1.0, 2.0, 3.0, 2.5, 2.0]})
    print(add_can_features(sample_df))
