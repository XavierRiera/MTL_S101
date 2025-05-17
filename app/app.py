'''import streamlit as st
import datetime
import streamlit.components.v1 as components
import base64
from io import BytesIO

# Declare the local Streamlit audio recorder component
st_audiorec = components.declare_component(
    "st_audiorec",
    path="app/streamlit-audio-recorder/st_audiorec/frontend/build"
)

# Page setup
st.set_page_config(page_title="Birdify - Bird Sound Classifier", layout="wide")

# Custom styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #5BA6B1;
        font-size: 20px;
    }
    .stButton > button {
        font-size: 20px !important;
        padding: 0.5em 1em;
    }
    .stTextInput input {
        font-size: 18px !important;
    }
    .stFileUploader label {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display logos
col_logo1, col_logo2, _ = st.columns([1, 1, 6])
with col_logo1:
    st.image("https://www.upf.edu/image/company_logo?img_id=10601&t=1718038903187", width=150)
with col_logo2:
    st.image("app/images/birdify-logo.png", width=150)

# Page title and description
st.markdown("<br>", unsafe_allow_html=True)
st.title("🐦 Birdify")
st.markdown("#### Identify bird species from audio recordings using machine learning!")

# File upload and audio recording
col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 Upload Audio File")
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

with col2:
    st.subheader("🎙️ Record Audio")
    st.markdown("Click the button to start and stop recording.")
    wav_audio_data = st_audiorec()

# Classification logic
st.markdown("---")
if st.button("🔍 Classify"):
    if uploaded_file:
        st.success("✅ Audio uploaded. Predicted Bird Species: European Robin")
        st.audio(uploaded_file)

    elif wav_audio_data:
        st.write("DEBUG — Recorded object:", wav_audio_data)  # 👈 Add this line

        # Try to extract base64 if present
        if isinstance(wav_audio_data.get("arr"), str) and "," in wav_audio_data["arr"]:
            header, b64data = wav_audio_data["arr"].split(",")
            audio_bytes = base64.b64decode(b64data)
            audio_buffer = BytesIO(audio_bytes)

            st.success("✅ Audio recorded. Predicted Bird Species: Eurasian Blackbird")
            st.audio(audio_buffer, format="audio/wav")
        else:
            st.error("⚠️ Recording returned unexpected format. Please try again.")

    else:
        st.error("⚠️ Please upload or record an audio file before classifying.")

# Footer
st.markdown("---")
st.caption(f"Birdify Project • {datetime.datetime.now().year}")
'''

import streamlit as st
from audio_utils import extract_features
from model_loader import load_keras_model, load_pickle_model
from app.prediction import predict_with_model
from io import BytesIO
import tempfile
import numpy as np

st.set_page_config(page_title="Birdify - Bird Classifier", layout="centered")
st.title("🐦 Birdify - Bird Sound Classifier")
st.markdown("Upload a bird sound (.wav) to classify the species using a trained ML model.")

uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])
model_choice = st.selectbox("Choose a model", [
    "cnn_trained.h5", 
    "cnn_lstm_trained.h5", 
    "knn_trained.h5", 
    "svm_rbf_trained.h5", 
    "best_stacked_model.pkl"
])

model_type = "keras" if model_choice.endswith(".h5") else "sklearn"

if st.button("🔍 Classify"):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.audio(uploaded_file)

        features = extract_features(tmp_path)
        if features is not None:
            if model_type == "keras":
                model = load_keras_model(f"models/{model_choice}")
            else:
                model = load_pickle_model(f"models/{model_choice}")

            prediction = predict_with_model(model, features, model_type=model_type)
            st.success(f"🎉 Predicted class index: {prediction}")
        else:
            st.error("Feature extraction failed. Check the audio file.")
    else:
        st.warning("Please upload an audio file.")