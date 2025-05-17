import librosa
import numpy as np

def extract_features(file, sr=22050, duration=5, offset=0.5, n_mfcc=40):
    y, sr = librosa.load(file, sr=sr, duration=duration, offset=offset)
    if len(y) < 1:
        return None
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled.reshape(1, -1)