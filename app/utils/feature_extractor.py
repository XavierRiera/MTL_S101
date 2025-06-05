'''import numpy as np
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
        return None'''
#opt2
'''import numpy as np
import essentia.standard as ess
from pathlib import Path

EXPECTED_NUM_FEATURES = 194  # Adjust to match your model if needed

# Initialize once and reuse to speed up inference
extractor = ess.FreesoundExtractor(
    lowlevelStats=["mean", "stdev"],
    tonalStats=["mean", "stdev"],
    mfccStats=["mean", "stdev"],
    gfccStats=["mean", "stdev"],
    lowlevelFrameSize=4096,
    lowlevelHopSize=2048,
    lowlevelSilentFrames="drop"
)

def extract_features(file_path):
    """
    Extracts a fixed-length vector of features from a single audio file
    using Essentia (same setup as training pipeline).
    Returns a 2D array: shape (1, n_features)
    """
    try:
        # Run the extractor
        features, _ = extractor(str(file_path))

        # Filter only the same numerical descriptors used during training
        feature_names = sorted([
            desc for desc in features.descriptorNames()
            if isinstance(features[desc], (float, int)) and
            any(x in desc for x in ["lowlevel", "mfcc", "gfcc", "tonal"])
        ])

        values = np.array([features[name] for name in feature_names], dtype=np.float32)

        return values.reshape(1, -1)

    except Exception as e:
        print(f"Feature extraction failed for {file_path}: {e}")
        return None'''

import numpy as np
from pathlib import Path
import essentia.standard as ess

# Constants
EXPECTED_NUM_FEATURES = 194  # Adjusted based on your training pipeline
FEATURE_PREFIXES = ["lowlevel", "mfcc", "gfcc", "tonal"]

# Use the same configuration as in training
extractor = ess.FreesoundExtractor(
    lowlevelStats=["mean", "stdev", "dmean", "dvar"],
    tonalStats=["mean", "stdev"],
    mfccStats=["mean", "stdev"],
    gfccStats=["mean", "stdev"],
    lowlevelFrameSize=4096,
    lowlevelHopSize=2048,
    lowlevelSilentFrames="drop"
)

def extract_features(filepath):
    """
    Extract a consistent feature vector (1, N) for model prediction.
    
    Args:
        filepath (str): Path to .wav file.
    
    Returns:
        np.ndarray: Array of shape (1, N) with numerical features.
    """
    try:
        # Run the feature extractor
        features, _ = extractor(str(filepath))

        # Select numeric features used in training
        selected = sorted([
            name for name in features.descriptorNames()
            if isinstance(features[name], (float, int)) and
               any(name.startswith(prefix) for prefix in FEATURE_PREFIXES)
        ])

        values = [features[name] for name in selected]
        return np.array(values).reshape(1, -1)

    except Exception as e:
        print(f"[ERROR] Feature extraction failed for {filepath}: {e}")
        return None
