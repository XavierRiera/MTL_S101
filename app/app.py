import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import time

from sklearn.preprocessing import normalize

from utils.feature_extractor import extract_features
from utils.model_loader import load_pickle_model

# Path configuration - using Path for cross-platform compatibility
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH_SVM = BASE_DIR / "models" / "svm_model_package.pkl"
MODEL_PATH_KNN = BASE_DIR / "models" / "knn_model_package.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"  
BIRD_INFO_PATH = BASE_DIR / "app" / "info_species.csv"

IMAGE_PATH = BASE_DIR / "app" / "birdify-logo3.png"  # Adjust if needed

def load_svm_model():
    model_package = joblib.load(MODEL_PATH_SVM)
    return (
        model_package['model'], 
        model_package['label_encoder'],
        model_package.get('scaler', None)  # Handle if scaler exists
    )


def load_knn_model():
     model_package = joblib.load(MODEL_PATH_KNN)
     return (
         model_package['model'], 
         model_package['label_encoder'],
         model_package.get('scaler', None),  # Handle if scaler exists
         #model_package.get('best_params', None)  # Include best params for reference
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
    **Birdify** is a machine learning-powered web application that classifies bird species based on audio input using two different AI models. It aims to combine ecological awareness with cutting-edge AI in a simple, user-friendly tool.

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
    - 🌱 **Raise biodiversity awareness** through interactive bird sound recognition
    - 🧠 **Compare machine learning approaches** with dual-model implementation
    - 🎙️ **Support both real-time recording and file uploads** for audio input
    - 🧪 **Facilitate education and research** on bird populations through accessible tech

    ---

    ### 🔬 How It Works
    1. Upload or record a `.wav` audio file
    2. The app extracts 20 key audio features using Essentia
    3. Features are classified using either:
       - **SVM Model**: Support Vector Machine (higher accuracy)
       - **KNN Model**: k-Nearest Neighbors (faster processing)
    4. View top 3 predicted species with detailed information

    ---

    ### 🧰 Technology Stack

    - **Frontend**: Streamlit (interactive web UI)
    - **Audio Processing**: 
      - **Essentia**: 20 spectral features extraction
      - **Librosa**: Backup MFCC features
    - **Machine Learning**:
      - **Scikit-learn**: SVM and KNN implementations
      - **Model Comparison**: Side-by-side accuracy testing
    - **Data Handling**: 
      - Pandas for species information
      - Joblib for model serialization
    - **Dataset**: 114 Bird Species with curated audio samples

    ---

    ### 🦉 Model Comparison
    | Feature        | SVM Model          | KNN Model          |
    |---------------|--------------------|--------------------|
    | Accuracy      | Higher (~92%)      | Good (~85%)        |
    | Speed         | Slower             | Faster             |
    | Best For      | Precise ID         | Quick analysis     |
    | Features      | Spectral analysis  | MFCC + temporal    |

    ---

    ### 🚀 Key Features
    - **Dual-model system** for comparison
    - **Top 3 predictions** with confidence scores
    - **Detailed species profiles**:
      - Conservation status
      - Habitat maps
      - Behavioral information
    - **Interactive audio player** for playback

    ---

    ### 🎓 Educational Value
    Birdify serves as both a field tool and educational platform, demonstrating:
    - How machine learning interprets bird sounds
    - Differences between classification algorithms
    - Importance of acoustic biodiversity

    📌 *Developed for "Taller de Tecnologia Musical" at [UPF](https://www.upf.edu)*  
    """)
elif page == "Bird Sound Classifier":
    st.title("🐦 Bird Sound Classifier")
    # Popup with quick start guide
    if st.button("🚀 Quick Start Guide"):
        with st.expander("How to Use Birdify - Quick Start", expanded=True):
            st.markdown("""
            **1. Upload Audio**  
            ➡ Click 'Browse files' or drag & drop a .wav file  
            
            **2. Select Model**  
            🔍 Choose between SVM (more precise) or KNN (faster)  
            
            **3. Classify**  
            🎯 Click the 'Classify' button  
            
            **4. View Results**  
            🐦 See top 3 predictions with detailed bird information  
            
            **5. Explore**  
            🔗 Click Wikipedia links for more species information  
            """)
    
    st.markdown("Upload a bird call recording to identify the species")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

    # Add model selection dropdown
    model_option = st.selectbox(
        "Select Classification Model",
        ("SVM Model", "KNN Model"),
        index=0
    )
    
    if st.button("🔍 Classify"):
        if uploaded_file is not None:
            with st.spinner("Processing audio..."):    
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                time.sleep(1)
            
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
                    
                    # Create a DataFrame for better visualization
                    features_df = pd.DataFrame(features_array, columns=TARGET_FEATURES)
                    
                    # Load model and encoder
                    try:
                        if model_option == "SVM Model":
                            model, label_encoder, scaler = load_svm_model()
                        else:  # KNN Model
                            model, label_encoder, scaler = load_knn_model()
                        
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
                        #st.write("### Prediction Probabilities")
                        proba_df = pd.DataFrame({
                            'Species': label_encoder.classes_,
                            'Probability': prediction_proba[0]
                        }).sort_values('Probability', ascending=False)
                        #st.dataframe(proba_df)
                        
                        top_preds = proba_df[proba_df['Probability'] > 0].head(3)
                            
                        if len(top_preds) == 0:
                            st.warning("No predictions with probability > 0 found")
                        else:
                            st.success(f"Analysis complete using {model_option}!")
                            st.markdown("## 🐤 Top Predicted Birds")
                            
                            for i, (_, row) in enumerate(top_preds.iterrows(), 1):
                                species = row['Species']
                                prob = row['Probability']
                                confidence_text = "✅ Very likely" if prob >= 0.85 else (
                                                "🔶 Moderate confidence" if prob >= 0.5 else "⚠️ Low confidence")
                                
                                bird_row = bird_info_df[bird_info_df["name"] == species]
                                
                                if not bird_row.empty:
                                    bird = bird_row.iloc[0]
                                    
                                    st.markdown(f"### {i}. {bird['name'].replace('_', ' ')} (_{bird['scientific_name']}_)")
                                    st.markdown(f"**Confidence**: {prob:.2%} ({confidence_text})")
                                    
                                    # Create columns for layout
                                    img_col, info_col = st.columns([1, 2])
                                    
                                    with img_col:
                                        if pd.notna(bird["image"]):
                                            st.image(bird["image"], width=250)
                                        if pd.notna(bird["conservation_status"]):
                                            st.image(bird["conservation_status"], caption="Conservation Status", width=250)
                                        if pd.notna(bird["map_image"]) and bird["map_image"] != "-":
                                            st.image(bird["map_image"], caption="Distribution Map", width=250)
                                    
                                    with info_col:
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
                                else:
                                    st.warning(f"No information available for species: {species}")
                                    st.markdown("---")

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