import joblib
from tensorflow.keras.models import load_model

def load_keras_model(path):
    return load_model(path)

def load_pickle_model(path):
    return joblib.load(path)