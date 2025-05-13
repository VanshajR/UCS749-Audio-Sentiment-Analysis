# Audio Emotion Recognition System

This project implements a real-time audio emotion recognition system using machine learning. It can analyze speech and predict the emotional state of the speaker.

## Features
- Real-time audio emotion recognition
- Support for uploaded audio files
- Visualization of audio features
- Support for multiple emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised

## Dataset
This project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech) dataset. You can download it from:
https://zenodo.org/record/1188976

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

3. Download the RAVDESS dataset and place it in the `data` folder

4. Run the application:
```bash
streamlit run app.py
```

## Project Structure
```
├── app.py                 # Main Streamlit application
├── model.py              # Model training and prediction code
├── audio_processor.py    # Audio processing utilities
├── data/                 # Dataset directory
├── models/              # Saved model files
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
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