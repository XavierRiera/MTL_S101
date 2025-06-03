import streamlit as st
import datetime
import streamlit.components.v1 as components
import base64
from io import BytesIO
import os
import tempfile
import numpy as np
import pandas as pd

from utils.feature_extractor import extract_features
from utils.model_loader import load_pickle_model
from prediction import predict_with_model

# Declare the local Streamlit audio recorder component
st_audiorec = components.declare_component(
    "st_audiorec",
    path="streamlit-audio-recorder/st_audiorec/frontend/build"
)

# Sidebar navigation
st.set_page_config(page_title="Birdify - Bird Classifier", layout="wide")
st.sidebar.image("https://www.upf.edu/image/company_logo?img_id=10601&t=1718038903187", width=200)
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Bird Sound Classifier", "About Project"])

st.image("data/images/birdify-logo3.png", width=400)

#Design html
st.markdown("""
    <style>

    .stApp {
        background-color: #e4edfe;
        font-family: 'Segoe UI', sans-serif;
        color: #2a2a2a;
    }

    .block-container {
        padding: 2rem 4rem;
    }

    h1, h2, h3 {
        color: #2a5d8f;
        font-weight: 700;
    }

    .stSidebar {
        background-color: #e3f2fd !important;
        color: #2a5d8f;
    }

    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar p {
        color: #2a5d8f;
    }

    .stButton > button {
        background-color: #5ba6b1;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 12px;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background-color: #468d97;
        color: white;
    }

    .stTextInput > div > input,
    .stFileUploader > label,
    .stSelectbox > div,
    .stMarkdown {
        font-size: 16px !important;
        color: #333333;
    }

    img {
        border-radius: 12px;
        border: 1px solid #d0e3ec;
    }

    .stCaption {
        font-style: italic;
        color: #6e6e6e;
    }

    hr {
        border: none;
        border-top: 1px solid #cccccc;
        margin: 2rem 0;
    }

    a {
        color: #327ba8 !important;
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
    }

    </style>
""", unsafe_allow_html=True)

# Paths
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_stacked_model.pkl"))
ENCODER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks", "models", "label_encoder.pkl"))

if page == "About Project":
    st.title("📘 About Birdify")
    st.markdown("""
    **Birdify** is a machine learning-powered web application that allows users to identify bird species through audio recordings. The core goals of the project are:

    - 🌱 **Promote biodiversity awareness** by making bird sound recognition accessible.
    - 🧠 **Apply machine learning** models trained on extracted audio features using Essentia.
    - 🎙️ **Enable real-time or uploaded audio input** for flexibility.
    - 🧪 Support scientific and educational applications through an intuitive interface.

    ### How It Works
    1. You can either upload a `.wav` file or record directly using your microphone.
    2. The app extracts numerical audio features from the sound.
    3. A pre-trained classification model (SVM or hybrid) predicts the most likely bird species.
    4. If supported, confidence scores and bird details are shown.

    ### Technology Stack
    - **Frontend**: Streamlit
    - **Audio Processing**: Librosa + Essentia
    - **ML Models**: scikit-learn, Keras, hybrid approaches
    - **Data**: Public datasets of bird calls

    ---
    📂 This tool is part of a final project of the usbject "Taller de Tecnologia Musical" at [UPF](https://www.upf.edu) to support applied AI in sound classification.
    """)

elif page == "Bird Sound Classifier":
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
                    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "info_species.csv"))
                    bird_info_df = pd.read_csv(csv_path)
                    model = load_pickle_model(MODEL_PATH)
                    label_encoder = load_pickle_model(ENCODER_PATH)
                    

                    # Show class probabilities if available
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(features)[0]
                        # Sort top 3 predictions with prob > 0
                        top_index = np.argmax(probs)
                        species = label_encoder.inverse_transform([top_index])[0]
                        
                        top_indices = np.argsort(probs)[::-1][:3]
                        top_preds = [(label_encoder.inverse_transform([i])[0], probs[i]) for i in top_indices if probs[i] > 0]

                        st.markdown("## 🐤 Top Predicted Birds")

                        for species, prob in top_preds:
                            # Format confidence level
                            if prob >= 0.85:
                                confidence_text = "✅ Very likely"
                            elif prob >= 0.5:
                                confidence_text = "🔶 Moderate confidence"
                            else:
                                confidence_text = "⚠️ Low confidence"

                            bird_row = bird_info_df[bird_info_df["name"] == species.replace(" ", "_")]

                            if bird_row.empty:
                                st.warning(f"No metadata found for **{species}**")
                                continue

                            bird = bird_row.iloc[0]

                            # Begin layout in two columns
                            img_col, info_col = st.columns([1, 2])

                            with img_col:
                                if pd.notna(bird["image"]):
                                    st.image(bird["image"], width=350)

                                if pd.notna(bird["conservation_status"]):
                                    st.image(bird["conservation_status"], caption="Conservation Status", width=350)

                                if pd.notna(bird["map_image"]) and bird["map_image"] != "-":
                                    st.image(bird["map_image"], caption="Distribution Map", width=300)

                            with info_col:
                                st.markdown(f"### **{bird['name'].replace('_', ' ')}** (_{bird['scientific_name']}_)")
                                st.markdown(f"**🔎 Confidence**: {prob:.2%} ({confidence_text})")
                                st.markdown(f"**Genus**: {bird['genus']}  |  **Species**: {bird['species']}")
                                st.markdown(f"**Habitat**: {bird['habitat']}")
                                st.markdown(f"**Diet**: {bird['diet']}")
                                st.markdown(f"**Behavior**: {bird['behavior']}")
                                st.markdown(f"**Distribution**: {bird['distribution']}")
                                st.markdown(f"**Reproduction**: {bird['reproduction']}")
                                st.markdown(f"**Uses**: {bird['uses']}")
                                st.markdown(f"**Size**: Height: {bird['height']} | Weight: {bird['weight']}")
                                st.markdown(f"**Description**: {bird['description']}")
                                st.markdown(f"[🔗 Learn more on Wikipedia]({bird['url']})", unsafe_allow_html=True)

                            st.markdown("---")


                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    
            else:
                st.warning("⚠️ Feature extraction failed. Please check the audio file.")
        else:
            st.error("⚠️ Please upload or record an audio file before classifying.")

    st.markdown("---")
    st.caption(f"Birdify Project • {datetime.datetime.now().year}")

