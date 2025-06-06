import essentia.standard as ess
from pathlib import Path
import numpy as np

# List of the 20 specific features we want to extract
TARGET_FEATURES = [
'lowlevel.spectral_energyband_middle_low.stdev','lowlevel.hfc.stdev',	'lowlevel.pitch_salience.stdev',	
'lowlevel.spectral_centroid.mean',	'lowlevel.spectral_energyband_high.stdev',	'lowlevel.spectral_complexity.mean',	'lowlevel.spectral_decrease.stdev',
'lowlevel.spectral_strongpeak.stdev',	'lowlevel.spectral_complexity.stdev',	'lowlevel.spectral_energyband_middle_low.mean',	'lowlevel.spectral_strongpeak.mean',
'lowlevel.loudness_ebu128.integrated',	'lowlevel.erbbands_skewness.mean',	'lowlevel.pitch.mean',	'lowlevel.spectral_flux.stdev',	'lowlevel.spectral_rolloff.mean',
'lowlevel.zerocrossingrate.mean', 'lowlevel.silence_rate_60dB.stdev',	'lowlevel.spectral_energyband_middle_high.mean',	'lowlevel.pitch_salience.mean'
]

def initialize_extractor():
    """Configure audio feature extractor with optimal settings for bird calls"""
    return ess.FreesoundExtractor(
        lowlevelStats=["mean", "stdev"],
        tonalStats=["mean", "stdev"],
        lowlevelFrameSize=2048,  # Smaller window for bird calls
        lowlevelHopSize=1024,
        lowlevelSilentFrames="drop"
    )

def extract_features(audio_path):
    """
    Extract the 20 target features from a WAV file
    
    Args:
        audio_path (str or Path): Path to the WAV file to process
        
    Returns:
        dict: Dictionary of feature names and their values
        None: If extraction fails
    """
    # Convert to Path object if it isn't already
    audio_path = Path(audio_path) if not isinstance(audio_path, Path) else audio_path
    
    # Verify file exists
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found at {audio_path}")
    
    # Initialize extractor
    extractor = initialize_extractor()
    
    try:
        # Extract all features
        features, _ = extractor(str(audio_path))
        
        # Extract only our target features
        feature_values = {}
        for feature_name in TARGET_FEATURES:
            try:
                feature_values[feature_name] = float(features[feature_name])
            except KeyError:
                # If a feature is missing, use NaN (model should handle this)
                feature_values[feature_name] = np.nan
                print(f"Warning: Feature {feature_name} not found in extraction")
        
        return feature_values
    
    except Exception as e:
        print(f"Error processing {audio_path.name}: {str(e)}")
        return None