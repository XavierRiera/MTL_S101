import joblib
from tensorflow.keras.models import load_model

def load_keras_model(path):
    return load_model(path)

def load_pickle_model(path):
    model_object = joblib.load(path)
    if isinstance(model_object, dict) and "model" in model_object:
        return model_object["model"]
    return model_object
