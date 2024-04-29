import streamlit as st
import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
import pyttsx3

# Function to generate caption
def generate_caption(video_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained processor, tokenizer, and model
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

    # Load video
    container = av.open(video_path)

    # Extract evenly spaced frames from video
    seg_len = container.streams.video[0].frames
    clip_len = model.config.encoder.num_frames
    indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))

    # Generate caption
    gen_kwargs = {
        "min_length": 10,
        "max_length": 50,
        "num_beams": 8,
    }
    pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
    tokens = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    return caption

# Function to convert text to speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.say(text)
    engine.runAndWait()

def main():
    st.title("Video Captioning with Audio")

    # Upload video file
    video_file = st.file_uploader("Upload video file", type=["mp4"])

    if video_file is not None:
        st.video(video_file)

        # Generate caption and audio
        if st.button("Generate Caption and Audio"):
            # Save uploaded video locally
            with open("uploaded_video.mp4", "wb") as f:
                f.write(video_file.read())

            # Generate caption
            caption = generate_caption("uploaded_video.mp4")

            # Display caption
            st.write("Caption:", caption)

            # Convert caption to audio and play
            text_to_speech(caption)

if __name__ == "__main__":
    main()
