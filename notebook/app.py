import torch
from torch.utils.model_zoo import load_url
from scipy.special import expit
import librosa
import numpy as np
from moviepy.editor import VideoFileClip
import tempfile
import streamlit as st
import os
import urllib.request
from io import BytesIO
import yt_dlp
import sys
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'blazeface')))
from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet, weights
from isplutils import utils

# Model and configuration settings
net_model = 'EfficientNetAutoAttB4'
train_db = 'DFDC'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
face_policy = 'scale'
face_size = 224
frames_per_video = 64

# Load the deepfake detection model
def load_deepfake_model():
    model_url = weights.weight_url[f'{net_model}_{train_db}']
    net = getattr(fornet, net_model)().eval().to(device)
    net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))
    return net

net = load_deepfake_model()

# Load BlazeFace for face detection
def load_face_detector():
    facedet = BlazeFace().to(device)
    # Use the correct relative paths or update to absolute paths as necessary
    facedet.load_weights(os.path.join(os.path.dirname(__file__), "../blazeface/blazeface.pth"))
    facedet.load_anchors(os.path.join(os.path.dirname(__file__), "../blazeface/anchors.npy"))
    return facedet

facedet = load_face_detector()
transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

videoreader = VideoReader(verbose=False)
video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)

# Progress Bar function
def show_progress(progress, total_steps, description="Processing"):
    st.progress(progress / total_steps)
    st.text(f"{description}: {progress}/{total_steps}")

# Function to extract video from URL
def extract_video_from_url(url):
    try:
        ydl_opts = {
            'format': 'best',
            'quiet': True,
            'outtmpl': tempfile.mktemp(suffix='.mp4')
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_url = info_dict['url']

            # Read video directly from URL
            response = urllib.request.urlopen(video_url)
            video_data = response.read()
            video_buffer = BytesIO(video_data)
            return video_buffer

    except Exception as e:
        st.error(f"Error extracting video from URL: {e}")
        return None

# Function to extract audio from video
def extract_audio_from_video(video_path):
    try:
        with VideoFileClip(video_path) as video_clip:
            audio_path = tempfile.mktemp(suffix='.wav')
            video_clip.audio.write_audiofile(audio_path)
            return audio_path
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None

# Analyze audio for fakeness score (placeholder logic)
def analyze_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        audio_score = np.random.rand()  # Placeholder for actual model inference
        return audio_score
    except Exception as e:
        st.error(f"Error in audio analysis: {e}")
        return None

# Process video frames and faces
def process_video_frames(video_file):
    try:
        with VideoFileClip(video_file) as video_clip:
            duration = video_clip.duration
            frame_rate = video_clip.fps
            total_frames = int(duration * frame_rate)

            vid_faces = face_extractor.process_video(video_file)

            if not vid_faces or all(len(frame['faces']) == 0 for frame in vid_faces):
                st.error("No faces detected in the video.")
                return None, None, None, None

            valid_faces = [frame['faces'][0] for frame in vid_faces if len(frame['faces']) > 0]
            if not valid_faces:
                st.error("No valid faces detected in the video.")
                return None, None, None, None

            faces_t = torch.stack([transf(image=face)['image'] for face in valid_faces])

            return faces_t, duration, frame_rate, total_frames

    except Exception as e:
        st.error(f"Error processing video frames: {e}")
        return None, None, None, None

# Run deepfake detection model on the faces
def analyze_faces(faces_t):
    try:
        with torch.no_grad():
            faces_pred = net(faces_t.to(device)).cpu().numpy().flatten()
        fake_score = expit(faces_pred.mean())
        return fake_score
    except Exception as e:
        st.error(f"Error analyzing faces: {e}")
        return None

# Create downloadable CSV report
def create_report(fake_score, audio_score, duration, frame_rate, total_frames):
    report_data = {
        'Fake Score': [fake_score],
        'Audio Score': [audio_score],
        'Duration': [duration],
        'Frame Rate': [frame_rate],
        'Total Frames': [total_frames]
    }
    df = pd.DataFrame(report_data)
    return df.to_csv(index=False)

# Main function to process video and display results
def process_video(video_file):
    try:
        # Show progress step by step
        show_progress(1, 5, "Extracting video frames")
        faces_t, duration, frame_rate, total_frames = process_video_frames(video_file)

        if faces_t is None:
            return

        show_progress(2, 5, "Analyzing faces")
        # Analyze faces using deepfake model
        fake_score = analyze_faces(faces_t)

        show_progress(3, 5, "Extracting audio")
        # Extract and analyze audio
        audio_path = extract_audio_from_video(video_file)
        audio_score = analyze_audio(audio_path) if audio_path else None

        show_progress(4, 5, "Generating report")
        # Create downloadable report
        csv_report = create_report(fake_score, audio_score, duration, frame_rate, total_frames)

        # Display results
        st.success(f"Video Fakeness Score: {fake_score:.4f}")
        st.text(f"Audio Fakeness Score: {audio_score:.4f}" if audio_score is not None else "Audio Fakeness Score: N/A")
        st.text(f"Duration: {duration:.2f} sec")
        st.text(f"Frame Rate: {frame_rate} FPS")
        st.text(f"Total Frames: {total_frames}")

        # Downloadable report
        st.download_button(
            label="Download Report as CSV",
            data=csv_report,
            file_name="deepfake_report.csv",
            mime="text/csv"
        )

        # Decision logic: Determine if the video is real or fake based on the fakeness score
        if fake_score > 0.5:
            st.error("The video is suspected to be FAKE.")
        else:
            st.success("The video is suspected to be REAL.")

        show_progress(5, 5, "Completed")

    except Exception as e:
        st.error(f"Error processing video: {e}")

# Streamlit application
def streamlit_app():
    st.title("Deepfake Detection System")

    # Upload a local video
    uploaded_file = st.file_uploader("Upload a video (mp4/avi/mov/mkv)", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        temp_video_path = tempfile.mktemp(suffix=".mp4")
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        if st.button("Analyze Uploaded Video"):
            st.info("Processing uploaded video...")
            process_video(temp_video_path)

    # URL input
    video_url = st.text_input("Enter video URL (from YouTube, Twitter, etc.)")
    if video_url:
        if st.button("Analyze Video from URL"):
            st.info("Processing video from URL...")
            video_buffer = extract_video_from_url(video_url)
            if video_buffer:
                temp_video_path = tempfile.mktemp(suffix='.mp4')
                with open(temp_video_path, 'wb') as temp_file:
                    temp_file.write(video_buffer.read())
                process_video(temp_video_path)

if __name__ == "__main__":
    streamlit_app()
