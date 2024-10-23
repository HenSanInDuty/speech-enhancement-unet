import asyncio
import io
import copy
import streamlit as st
import os
import librosa
import tempfile
import soundfile as sf
import numpy as np
from util import prediction_denoise

async def main():
    st.title(':musical_note: Speech Enhancement')
    st.subheader('Remove your audio background noise using Artificial Intelligence')
    
    # Input file
    input = st.file_uploader("Upload your audio")
    
    if st.button("Denoise audio"):
        original_input = copy.copy(input)
        tmp = io.BytesIO(original_input.read())
        original_audio, original_sr = librosa.load(tmp, sr=8000)
        
        # Get duration of audio
        S = librosa.stft(original_audio)
        min_duration = librosa.get_duration(S=S, sr=original_sr)
        
        # Inference
        working_path = os.getcwd()
        weights_path = os.path.join(working_path, "model", "unet")
        name_model = "mod_unet_last_weights"
        frame_length = original_audio.shape[0]
        hop_length_frame = original_audio.shape[0]
        min_duration = 2
        n_fft = 255
        hop_length_fft = round(original_audio.shape[0]/128)
        
        denoise, sr = prediction_denoise.prediction(weights_path, name_model, input, 8000, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft)

        col1, col2 = st.columns(2)
        
        with col1: 
            st.write("Original Audio")
            trim_audio, _ = librosa.effects.trim(original_audio)
            st.audio(trim_audio, sample_rate=8000)
            
        with col2:
            st.write("Denoise Audio")
            st.audio(denoise[0, :], sample_rate=sr)
            
        # Download session
        temp_audio = io.BytesIO()
        sf.write(temp_audio, denoise[0,:], sr, format='WAV')
        temp_audio.seek(0)
        
        st.download_button("Download Denoise Audio", temp_audio, f"audio_denoise.wav", mime="audio/wav")

if __name__ == '__main__':

    st.set_page_config(
        page_title="Noise Reduction",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    if __name__ == '__main__':
        asyncio.run(main())