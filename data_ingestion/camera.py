# data_ingestion/camera.py
import cv2

def load_video_frames(video_file):
    """Extract frames from a video file using OpenCV."""
    cap = cv2.VideoCapture(video_file)
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
    cap.release()
    return frames

if __name__ == "__main__":
    frames = load_video_frames("data/video.mp4")
    print(f"Total frames loaded: {len(frames)}")
