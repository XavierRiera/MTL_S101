import streamlit as st
#from streamlit_audio_recorder import audio_recorder
import datetime

st.set_page_config(page_title="Birdify - Bird Sound Classifier", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

logo_html = """
    <style>
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        height: 60px;  # You can adjust the height of the header as needed
        padding: 0 10px;
    }
    .header img {
        height: 40px;  # You can adjust the height of the logos as needed
    }
    </style>
    <div class="header">
        <div>
            <img src="https://www.upf.edu/image/company_logo?img_id=10601&t=1718038903187" alt="Logo UPF">
        </div>
    </div>
    """
st.markdown(logo_html, unsafe_allow_html=True)

st.title("🐦 Birdify")
st.markdown("Identify bird species from audio recordings using machine learning!")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📁 Upload Audio File")
    uploaded_file = st.file_uploader("Upload a .mp3 file", type=["mp3"])

with col2:
    st.subheader("🎙️ Record Audio")
    #recorded_audio = audio_recorder(text="Click to record", icon_size="2x")


#button classification
if st.button("🔍 Classify"):
    if uploaded_file:
        st.success("✅ Audio uploaded. Predicted Bird Species: European Robin")
        st.audio(uploaded_file)
    '''elif recorded_audio:
        st.success("✅ Audio recorded. Predicted Bird Species: Eurasian Blackbird")
        st.audio(recorded_audio, format="audio/wav")'''
else:
        st.warning("Please upload or record an audio file to classify.")

# Footer
st.caption(f"Birdify Project • {datetime.datetime.now().year}")