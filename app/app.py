import streamlit as st
import datetime
#from streamlit_audio_recorder import audio_recorder

st.set_page_config(page_title="Birdify - Bird Sound Classifier", layout="centered")
st.write("🧪 App loaded successfully")

# Custom color styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #5BA6B1;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p, .stButton {
        color: #5BA6B1;
    }
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
    }
    .header img {
        height: 45px;
        margin-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header with UPF + local Birdify logo
st.markdown(
    """
    <div class="header">
        <img src="https://www.upf.edu/image/company_logo?img_id=10601&t=1718038903187" alt="UPF Logo">
        <img src="app/images/birdify-logo.png" alt="Birdify Logo">
    </div>
    """,
    unsafe_allow_html=True
)

st.title("🐦 Birdify")
st.markdown("Identify bird species from audio recordings using machine learning!")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 Upload Audio File")
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

with col2:
    st.subheader("🎙️ Record Audio")
    recorded_audio = audio_recorder(text="Click to record", icon_size="2x")

# Classification logic
if st.button("🔍 Classify"):
    if uploaded_file:
        st.success("✅ Audio uploaded. Predicted Bird Species: European Robin")
        st.audio(uploaded_file)
    elif recorded_audio:
        st.success("✅ Audio recorded. Predicted Bird Species: Eurasian Blackbird")
        st.audio(recorded_audio, format="audio/wav")
    else:
        st.warning("Please upload or record an audio file to classify.")
else:
    st.info("Upload or record audio, then press 'Classify'.")

# Footer
st.caption(f"Birdify Project • {datetime.datetime.now().year}")

