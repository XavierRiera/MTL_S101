# Birdify 🐦 – Bird Sound Classification App

Birdify is a web-based app that allows users to identify bird species from audio recordings. Built using Streamlit, it supports both uploading `.wav` files and recording directly from the browser.

## 🎯 Features

- Upload `.wav` audio recordings for bird sound classification
- Record audio directly in-browser using `streamlit-audio-recorder`
- Real-time prediction (placeholder)
- Simple, elegant UI with Streamlit

## 🔧 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/XavierRiera/Birdify.git
   cd Birdify
   ```

2. (Optional but recommended) Create a virtual environment if not created previously:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r app/requirements.txt
   ```

4. Run the app:
   ```bash
   cd app
   streamlit run app.py
   ```

## 🔊 Audio Recorder

This app uses [`streamlit-audio-recorder`](https://github.com/stefanrmmr/streamlit-audio-recorder) to enable browser-based microphone recording.

## 📁 Folder Structure

```
Birdify/
├── app/
│   ├── app.py
│   └── images/
│       └── birdify-logo.png
├── requirements.txt
├── README.md
```

## ⚠️ Notes

- The classifier is a placeholder. You can integrate a model to classify the audio.
- Streamlit Cloud must allow mic access for recording to work.