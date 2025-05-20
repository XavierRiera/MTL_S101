def predict_with_model(model, features, model_type="keras"):
    if model_type == "keras":
        prediction = model.predict(features)
        predicted_class = prediction.argmax(axis=-1)[0]
    elif model_type == "sklearn":
        predicted_class = model.predict(features)[0]
    else:
        raise ValueError("Unsupported model type")
    return predicted_class