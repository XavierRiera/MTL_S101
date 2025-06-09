import joblib
import pickle

def load_pickle_model(path):
    try:
        # First try with UTF-8 encoding
        with open(path, 'rb') as f:
            model_object = joblib.load(f)
    except UnicodeDecodeError:
        # Fallback to latin-1 if UTF-8 fails
        with open(path, 'rb') as f:
            model_object = joblib.load(f, encoding='latin-1')
    
    if isinstance(model_object, dict) and "model" in model_object:
        return model_object["model"]
    return model_object
