import librosa
import numpy as np
import soundfile as sf
import os
from typing import Tuple, List
import sounddevice as sd

class AudioProcessor:
    def __init__(self, sample_rate: int = 22050, duration: int = 3):
        self.sample_rate = sample_rate
        self.duration = duration
        self.emotions = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }

    def record_audio(self, duration: int = None) -> Tuple[np.ndarray, int]:
        """Record audio from microphone."""
        if duration is None:
            duration = self.duration
            
        print(f"Recording for {duration} seconds...")
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1
        )
        sd.wait()
        return recording.flatten(), self.sample_rate

    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract features from audio file (matches notebook)."""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = mfccs.mean(axis=1)
        mfccs_std = mfccs.std(axis=1)
        # Delta MFCCs
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_mfccs_mean = delta_mfccs.mean(axis=1)
        delta_mfccs_std = delta_mfccs.std(axis=1)
        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        chroma_std = chroma.std(axis=1)
        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = contrast.mean(axis=1)
        contrast_std = contrast.std(axis=1)
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        tonnetz_mean = tonnetz.mean(axis=1)
        tonnetz_std = tonnetz.std(axis=1)
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = zcr.mean()
        zcr_std = zcr.std()
        # Root Mean Square Energy
        rms = librosa.feature.rms(y=audio)
        rms_mean = rms.mean()
        rms_std = rms.std()
        # Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        centroid_mean = centroid.mean()
        centroid_std = centroid.std()
        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        rolloff_mean = rolloff.mean()
        rolloff_std = rolloff.std()
        # Combine all features
        features = np.hstack([
            mfccs_mean, mfccs_std,
            delta_mfccs_mean, delta_mfccs_std,
            chroma_mean, chroma_std,
            contrast_mean, contrast_std,
            tonnetz_mean, tonnetz_std,
            [zcr_mean, zcr_std, rms_mean, rms_std, centroid_mean, centroid_std, rolloff_mean, rolloff_std]
        ])
        return features

    def save_audio(self, audio: np.ndarray, filename: str) -> str:
        """Save audio to file."""
        sf.write(filename, audio, self.sample_rate)
        return filename

    def get_emotion_from_filename(self, filename: str) -> str:
        """Extract emotion from RAVDESS filename."""
        emotion_code = filename.split('-')[2]
        return self.emotions.get(emotion_code, 'unknown')

    def process_audio_file(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process audio file and return waveform and features."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract features
            features = self.extract_features(audio_path)
            
            return audio, features
        except Exception as e:
            print(f"Error processing audio file: {str(e)}")
            return None, None

    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds."""
        try:
            return librosa.get_duration(path=audio_path)
        except Exception as e:
            print(f"Error getting audio duration: {str(e)}")
            return 0.0 