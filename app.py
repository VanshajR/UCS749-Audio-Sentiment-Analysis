import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import tempfile
from audio_processor import AudioProcessor
import sounddevice as sd
import soundfile as sf
import pickle
from tensorflow.keras.models import load_model

# Explicitly load the model and scaler from the models folder
MODEL_PATH = "models/emotion_model.h5"
SCALER_PATH = "models/scaler.pkl"

# Load model with debug prints
try:
    emotion_model_loaded = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    emotion_model_loaded = None

# Load scaler with debug prints
try:
    with open(SCALER_PATH, "rb") as f:
        scaler_loaded = pickle.load(f)
    print("Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler_loaded = None

# Define the emotion labels (make sure this matches your training)
EMOTIONS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def predict_emotion(features):
    if emotion_model_loaded is None or scaler_loaded is None:
        raise ValueError("Model or scaler not loaded.")
    print("Features shape:", features.shape)
    features_scaled = scaler_loaded.transform(features.reshape(1, -1))
    prediction = emotion_model_loaded.predict(features_scaled)[0]
    emotion_idx = prediction.argmax()
    confidence = prediction[emotion_idx]
    return EMOTIONS[emotion_idx], confidence

def predict_proba(features):
    if emotion_model_loaded is None or scaler_loaded is None:
        raise ValueError("Model or scaler not loaded.")
    features_scaled = scaler_loaded.transform(features.reshape(1, -1))
    prediction = emotion_model_loaded.predict(features_scaled)[0]
    return list(zip(EMOTIONS, prediction))

# Set page config
st.set_page_config(
    page_title="Audio Emotion Recognition",
    page_icon="🎵",
    layout="wide"
)

# Initialize audio processor
audio_processor = AudioProcessor()

def plot_waveform(audio, sr):
    """Plot audio waveform."""
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title('Audio Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    return plt

def plot_spectrogram(audio, sr):
    """Plot audio spectrogram."""
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    return plt

def main():
    st.title("🎵 Audio Emotion Recognition")
    st.write("Analyze emotions from speech using AI")

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])

    with tab1:
        st.header("Record Audio")
        duration = st.slider("Recording Duration (seconds)", 1, 10, 3)
        
        if st.button("Start Recording"):
            with st.spinner(f"Recording for {duration} seconds..."):
                # Record audio
                audio, sr = audio_processor.record_audio(duration)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    audio_processor.save_audio(audio, temp_file.name)
                    
                    # Audio preview
                    audio_bytes = open(temp_file.name, 'rb').read()
                    st.audio(audio_bytes, format='audio/wav')
                    
                    # Process and display results
                    display_results(temp_file.name)
                    
                # Clean up
                os.unlink(temp_file.name)

    with tab2:
        st.header("Upload Audio")
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                
                # Audio preview
                st.audio(uploaded_file, format='audio/wav')
                
                display_results(temp_file.name)
            os.unlink(temp_file.name)

def display_results(audio_path):
    """Display analysis results for the given audio file."""
    # Process audio
    audio, features = audio_processor.process_audio_file(audio_path)
    
    if audio is not None and features is not None:
        # Create columns for visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.pyplot(plot_waveform(audio, audio_processor.sample_rate))
        
        with col2:
            st.pyplot(plot_spectrogram(audio, audio_processor.sample_rate))
        
        # Get emotion prediction
        try:
            emotion, confidence = predict_emotion(features)
            
            # Display results
            st.success(f"Detected Emotion: {emotion.upper()}")
            st.info(f"Confidence: {confidence:.2%}")
            
            # Display probability distribution
            st.subheader("Emotion Probabilities")
            probs = predict_proba(features)
            
            # Create a bar chart of probabilities
            fig, ax = plt.subplots(figsize=(10, 4))
            emotions, probabilities = zip(*probs)
            ax.bar(emotions, probabilities)
            ax.set_title("Emotion Probability Distribution")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            print(f"Prediction error: {e}")
    else:
        st.error("Error processing audio file. Please try again.")

if __name__ == "__main__":
    main() 