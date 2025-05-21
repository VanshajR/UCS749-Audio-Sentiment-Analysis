# Audio Emotion Recognition System

This project implements a real-time audio emotion recognition system using machine learning. It can analyze speech and predict the emotional state of the speaker.

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://audio-sentiment-vanshajr.streamlit.app)

## Features
- Real-time audio emotion recognition
- Support for uploaded audio files
- Visualization of audio features
- Support for multiple emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised

## Dataset
This project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech) dataset. You can download it from:
`https://zenodo.org/record/1188976` with `Audio_Speech_Actors_01-24.zip` being used in particular.

### File naming convention

Each of the 7356 RAVDESS files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 02-01-06-01-02-01-12.mp4). These identifiers define the stimulus characteristics: 

#### Filename identifiers 

- Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
- Vocal channel (01 = speech, 02 = song).
- Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
- Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
- Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
- Repetition (01 = 1st repetition, 02 = 2nd repetition).
- Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

Filename example: 02-01-06-01-02-01-12.wav

Video-only (02)
Speech (01)
Fearful (06)
Normal intensity (01)
Statement "dogs" (02)
1st Repetition (01)
12th Actor (12)
Female, as the actor ID number is even.

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the RAVDESS dataset and place it in the `data` folder as a directory named `RAVDESS`

4. Run the application:
```bash
streamlit run app.py
```

## Project Structure
```
├── app.py                                     # Main Streamlit application
├── audio_emotion_recognition.ipynb            # Model training and prediction code
├── audio_processor.py                         # Audio processing utilities
├── data/RAVDESS                               # Dataset directory
├── models/                                    # Saved model files
└──  requirements.txt                          # Project dependencies

```

## Usage
1. Launch the application using `streamlit run app.py`
2. Choose between real-time recording or file upload
3. For real-time analysis, click "Start Recording" and speak
4. For file upload, select an audio file
5. View the emotion prediction and audio visualization

## Requirements
- Python 3.8+
- See requirements.txt for all dependencies 
