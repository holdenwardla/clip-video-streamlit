
import streamlit as st
import tempfile
import os
import torch
import clip
from PIL import Image
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_frames(video_path, every_n_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    count = 0
    success, image = cap.read()
    while success:
        if count % every_n_frames == 0:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
            frames.append(pil_img)
        success, image = cap.read()
        count += 1
    cap.release()
    return frames

def get_clip_embeddings(frames):
    embeddings = []
    for img in frames:
        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            embeddings.append(embedding.cpu().numpy()[0])
    return embeddings

st.title("🎥 Compare Any Two Videos with CLIP")

uploaded_files = st.file_uploader("Upload videos", type=["mp4", "mov", "avi"], accept_multiple_files=True)

video_paths = {}
if uploaded_files:
    st.info("Processing uploads...")
    for video in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            temp.write(video.read())
            video_paths[video.name] = temp.name

    names = list(video_paths.keys())
    vid1 = st.selectbox("Select Video 1", names, key="v1")
    vid2 = st.selectbox("Select Video 2", names, key="v2")

    if vid1 and vid2 and vid1 != vid2:
        st.info("Extracting frames...")
        frames1 = extract_frames(video_paths[vid1])
        frames2 = extract_frames(video_paths[vid2])

        if not frames1 or not frames2:
            st.error("Could not extract frames.")
        else:
            st.info("Generating embeddings...")
            emb1 = get_clip_embeddings(frames1)
            emb2 = get_clip_embeddings(frames2)

            st.success("Calculating similarity...")
            similarity_matrix = cosine_similarity(np.array(emb1), np.array(emb2))
            avg_score = np.mean(similarity_matrix)
            st.metric("Average Similarity Score", f"{avg_score:.4f}")

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(similarity_matrix, cmap="viridis", ax=ax)
            ax.set_title("Frame-to-Frame Similarity")
            ax.set_xlabel("Video 2 Frames")
            ax.set_ylabel("Video 1 Frames")
            st.pyplot(fig)

        os.remove(video_paths[vid1])
        os.remove(video_paths[vid2])
