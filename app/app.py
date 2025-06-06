import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.preprocessing import normalize

from utils.feature_extractor import extract_features
from utils.model_loader import load_pickle_model

# Path configuration - using Path for cross-platform compatibility
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "knn_model_package.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"  
BIRD_INFO_PATH = BASE_DIR / "app" / "info_species.csv"

IMAGE_PATH = BASE_DIR / "app" / "birdify-logo3.png"  # Adjust if needed

# def load_svm_model():
#     model_package = joblib.load(MODEL_PATH)
#     return (
#         model_package['model'], 
#         model_package['label_encoder'],
#         model_package.get('scaler', None)  # Handle if scaler exists
#     )


def load_knn_model():
    model_package = joblib.load(MODEL_PATH)
    return (
        model_package['model'], 
        model_package['label_encoder'],
        model_package.get('scaler', None),  # Handle if scaler exists
        model_package.get('best_params', None)  # Include best params for reference
    )


# Set page config
st.set_page_config(page_title="Birdify - Bird Classifier", layout="wide")

# Sidebar navigation
st.sidebar.image("https://www.upf.edu/image/company_logo?img_id=10601&t=1718038903187", width=200)
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Bird Sound Classifier", "About Project"])

# App logo
st.image(str(IMAGE_PATH), width=400)  # Convert Path to string

# CSS styling
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
    hr {
        border: none;
        border-top: 1px solid #cccccc;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)




# About page
if page == "About Project":
    st.title("📘 About Birdify")
    st.markdown("""
    **Birdify** is a machine learning-powered web application that classifies bird species based on audio input. It aims to combine ecological awareness with cutting-edge AI in a simple, user-friendly tool.

    ---
    ### 👥 Project Developers
    - Maheen Asad  
    - Cinta Carot  
    - Arnau Martín  
    - Zat Pros  
    - Xavier Riera  
    - Lluc Sayols  
    - Silvia Riaño  

    📂 GitHub repository: [github.com/XavierRiera/Birdify](https://github.com/XavierRiera/Birdify/tree/main)

    ---

    ### 🎯 Objectives
    - 🌱 **Raise biodiversity awareness** through interactive bird sound recognition.
    - 🧠 **Apply machine learning** techniques for acoustic classification.
    - 🎙️ **Support both real-time recording and file uploads** for audio input.
    - 🧪 **Facilitate education and research** on bird populations through accessible tech.

    ---

    ### 🔬 How It Works
    1. Upload or record a `.wav` audio file.
    2. The app extracts audio features using a preprocessing pipeline.
    3. The features are classified using a trained model (e.g., SVM or deep learning).
    4. Predicted bird species and confidence levels are displayed.

    ---

    ### 🧰 Technology Stack

    - **Frontend**: Streamlit (interactive web UI)
    - **Audio Processing**: Librosa + Essentia (feature extraction), NumPy (amplitude normalization), SciPy (filtering), noisereduce (denoising)
    - **Feature Extraction**:  
      - **Essentia**: spectral features, pitch, energy-based segmentation  
      - **Librosa**: MFCCs, chroma, mel-spectrogram  
      - **VGGish / OpenL3**: pre-trained feature embeddings (optional)
    - **ML Models**: Support Vector Machines (SVM), CNNs, hybrid approaches
    - **Backend Tools**: PyTorch, scikit-learn, Keras
    - **Dataset**: Kaggle - Sound of 114 Bird Species (segmented, pre-labeled)

    ---

    ### 🚀 Additional Features
    - Band-pass filtering to remove background and environmental noise
    - Wavelet-based denoising to preserve bird syllables
    - Real-time and batch audio input
    - Class balancing with augmentation: pitch/time shift, noise injection

    ---

    ### � Scientific & Educational Goals
    Birdify is inspired by existing tools like BirdNET and Wing Watch, aiming to support ecological monitoring, classroom learning, and hobbyist curiosity. The goal is to make AI in sound classification accessible and impactful.

    📌 *This tool is part of a final project for the course "Taller de Tecnologia Musical" at [UPF](https://www.upf.edu).*  
    """)
elif page == "Bird Sound Classifier":
    st.title("🐦 Bird Sound Classifier")
    st.markdown("Upload a bird call recording to identify the species")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
    
    if st.button("🔍 Classify"):
        if uploaded_file is not None:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Display audio player
            st.audio(uploaded_file)
            
            # Extract features
            try:
                features_dict = extract_features(tmp_path)
                
                if features_dict is None:
                    st.error("Feature extraction failed. Please try another audio file.")
                else:
                    # Get only the 20 target features in correct order
                    from utils.feature_extractor import TARGET_FEATURES
                    features_list = [features_dict[feature] for feature in TARGET_FEATURES]
                    features_array = np.array(features_list).reshape(1, -1)
                    
                    # DEBUG: Show feature statistics
                    st.write("### Feature Statistics")
                    st.write("Mean:", features_array.mean())
                    st.write("Min:", features_array.min())
                    st.write("Max:", features_array.max())
                    
                    # Create a DataFrame for better visualization
                    features_df = pd.DataFrame(features_array, columns=TARGET_FEATURES)
                    st.write("### Extracted Features")
                    st.dataframe(features_df)
                    
                    # Load model and encoder
                    try:

                        model, label_encoder, scaler, best_params = load_knn_model()
                        #model, label_encoder, scaler = load_svm_model()
                        
                        # DEBUG: Show model and encoder info
                        st.write("### Model Information")
                        st.write("Model type:", type(model))
                        st.write("Model parameters:", model.get_params())
                        st.write("Label encoder classes:", label_encoder.classes_)
                        
                        try:
                            bird_info_df = pd.read_csv(BIRD_INFO_PATH, encoding='utf-8')
                        except UnicodeDecodeError:
                            bird_info_df = pd.read_csv(BIRD_INFO_PATH, encoding='latin-1')
                        
                        if scaler:
                            features_scaled = scaler.transform(features_array)
                        else:
                            features_scaled = features_array  # Use as-is if no scaling needed

                        # SVM prediction (no kneighbors needed)
                        prediction_proba = model.predict_proba(features_scaled)
                        prediction = model.predict(features_scaled)
                                                
                        # DEBUG: Show prediction probabilities
                        st.write("### Prediction Probabilities")
                        proba_df = pd.DataFrame({
                            'Species': label_encoder.classes_,
                            'Probability': prediction_proba[0]
                        }).sort_values('Probability', ascending=False)
                        st.dataframe(proba_df)
                        
                        species = label_encoder.inverse_transform(prediction)[0]
                        
                        # Get bird info
                        bird_row = bird_info_df[bird_info_df["name"] == species]
                        
                        if not bird_row.empty:
                            bird = bird_row.iloc[0]
                            st.success(f"Predicted species: {bird['name'].replace('_', ' ')}")
                            
                            # Show top 3 predictions
                            top3 = proba_df.head(3)
                            st.write("### Top 3 Predictions")
                            for i, (_, row) in enumerate(top3.iterrows(), 1):
                                st.write(f"{i}. {row['Species'].replace('_', ' ')}: {row['Probability']:.2%}")
                        else:
                            st.warning(f"Species {species} found in predictions but not in our database")
                    
                    except Exception as e:
                        st.error(f"Model prediction failed: {str(e)}")
                        st.error("Full error traceback:")
                        st.exception(e)
            
            except Exception as e:
                st.error(f"Error processing audio file: {str(e)}")
                st.exception(e)
            
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
        else:
            st.warning("Please upload a WAV file first")