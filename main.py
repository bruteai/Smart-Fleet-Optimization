# main.py
from data_ingestion import can_bus, gps, camera, audio
from preprocessing import sync, feature_extraction

def run_pipeline():
    # Load raw sensor data
    can_df = can_bus.load_can_data("data/can_bus.csv")
    gps_df = gps.load_gps_data("data/gps.csv")
    frames = camera.load_video_frames("data/video.mp4")
    audio_feat = audio.load_audio_features("data/audio.wav")
    
    # Combine CAN and GPS data, then add additional features
    combined = sync.sync_data(can_df, gps_df)
    enhanced = feature_extraction.add_can_features(combined)
    
    # Save merged dataset for training
    enhanced.to_csv("data/merged_data.csv", index=False)
    print("Data pipeline completed.")

if __name__ == "__main__":
    run_pipeline()
