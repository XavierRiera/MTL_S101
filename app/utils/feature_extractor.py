import numpy as np
import librosa

EXPECTED_NUM_FEATURES = 194

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=5, offset=0.5)

        # MFCCs (40)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

        # Chroma (12)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

        # Mel Spectrogram (128)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

        # Spectral Contrast (7)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

        # Tonnetz (6)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)

        # Combine all
        features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])

        # Normalize to match model's expected input size (pad or trim)
        current_len = features.shape[0]
        if current_len < EXPECTED_NUM_FEATURES:
            # Pad with zeros
            padded = np.pad(features, (0, EXPECTED_NUM_FEATURES - current_len))
            return padded.reshape(1, -1)
        elif current_len > EXPECTED_NUM_FEATURES:
            # Trim excess
            trimmed = features[:EXPECTED_NUM_FEATURES]
            return trimmed.reshape(1, -1)
        else:
            return features.reshape(1, -1)

    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None