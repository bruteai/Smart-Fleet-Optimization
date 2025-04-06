# data_generation.py (Synthetic Data to test) > Use your own dataset, this is just sample!
import os
import pandas as pd
import numpy as np
import cv2
import wave
from datetime import datetime, timedelta

def create_can_csv(path, n=1000, start=None):
    if start is None:
        start = datetime.now()
    times = [start + timedelta(seconds=i) for i in range(n)]
    data = {
        'timestamp': times,
        'fuel_rate': np.random.uniform(1.0, 10.0, n),
        'speed': np.random.uniform(0, 100, n),
        'rpm': np.random.uniform(500, 4000, n),
        'engine_temp': np.random.uniform(70, 120, n),
        'vibration': np.random.uniform(0.1, 1.0, n)
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"CAN CSV created at {path}")

def create_gps_csv(path, n=1000, start=None):
    if start is None:
        start = datetime.now()
    times = [start + timedelta(seconds=i) for i in range(n)]
    base_lat, base_lon = 40.7128, -74.0060
    data = {
        'timestamp': times,
        'latitude': base_lat + np.random.normal(0, 0.0005, n),
        'longitude': base_lon + np.random.normal(0, 0.0005, n)
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"GPS CSV created at {path}")

def create_video_file(path, frames_count=100, width=640, height=480, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for _ in range(frames_count):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        video.write(frame)
    video.release()
    print(f"Video file created at {path}")

def create_audio_file(path, duration=5, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = np.random.normal(0, 0.05, sine_wave.shape)
    audio = sine_wave + noise
    audio_int = np.int16(audio / np.max(np.abs(audio)) * 32767)
    
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int.tobytes())
    print(f"Audio file created at {path}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    create_can_csv(os.path.join("data", "can_bus.csv"))
    create_gps_csv(os.path.join("data", "gps.csv"))
    create_video_file(os.path.join("data", "video.mp4"))
    create_audio_file(os.path.join("data", "audio.wav"))
    print("All synthetic data generated.")
