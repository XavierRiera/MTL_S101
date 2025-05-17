import librosa
import numpy as np

def extract_features(file_path, sr=22050, duration=5, offset=0.5, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=sr, duration=duration, offset=offset)
    if len(y) < 1:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, -1)
