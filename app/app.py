import streamlit as st
import datetime
import streamlit.components.v1 as components
import base64
from io import BytesIO
import os
import tempfile
import numpy as np

from utils.feature_extractor import extract_features
from utils.model_loader import load_pickle_model
from prediction import predict_with_model

from streamlit_audiorec import st_audiorec

# Declare the local Streamlit audio recorder component
st_audiorec = st_audiorec()

# Where the models are located
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_stacked_model.pkl"))
ENCODER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "models", "label_encoder.pkl"))

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
    st.image("images/birdify-logo.png", width=150)

# Page title and description
st.markdown("<br>", unsafe_allow_html=True)
st.title("🐦 Birdify")
st.markdown("#### Identify bird species from audio recordings using the best stacked model!")

# File upload and audio recording
col1, col2 = st.columns(2)
uploaded_file = None
wav_audio_data = None

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
    tmp_path = None
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
            st.audio(uploaded_file)

    elif wav_audio_data:
        if isinstance(wav_audio_data.get("arr"), str) and "," in wav_audio_data["arr"]:
            header, b64data = wav_audio_data["arr"].split(",")
            audio_bytes = base64.b64decode(b64data)
            audio_buffer = BytesIO(audio_bytes)
            st.audio(audio_buffer, format="audio/wav")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
        else:
            st.error("⚠️ Recording returned unexpected format.")
            tmp_path = None

    if tmp_path:
        features = extract_features(tmp_path)
        if features is not None:
            try:
                model = load_pickle_model(MODEL_PATH)
                label_encoder = load_pickle_model(ENCODER_PATH)
                
                #prediction
                prediction = model.predict(features)[0]
                species = label_encoder.inverse_transform([prediction])[0]
                st.success(f"🎉 Predicted bird species: **{species}**")

                # Show class probabilities if available
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]
                    st.markdown("#### 🔎 Confidence Scores")
                    for i, prob in enumerate(probs):
                        label = label_encoder.inverse_transform([i])[0]
                        st.write(f"**{label}**: {prob:.2%}")
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                
        else:
            st.warning("⚠️ Feature extraction failed. Please check the audio file.")
    else:
        st.error("⚠️ Please upload or record an audio file before classifying.")

# Footer
st.markdown("---")
st.caption(f"Birdify Project • {datetime.datetime.now().year}")