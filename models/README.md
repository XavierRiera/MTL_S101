## 🧠 Trained Models

This directory contains all the trained models used in **Birdify**, including classical machine learning models (like KNN and SVM) and deep learning models (CNN and AFCNN). These models were generated from the training and evaluation process described in the [`notebooks/`](../notebooks) directory.

Each file corresponds to a model or associated artifact (e.g., scalers or metadata) used for prediction and evaluation.

### 📁 Contents

| File                      | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `AFCNN.h5`                | Initial version of the Attention-based Fully Convolutional Neural Network. |
| `FINAL_AFCNN.h5`          | Final AFCNN model with optimized metrics and evaluation results.            |
| `TRAIN_AFCNN.h5`          | AFCNN model saved right after training, before post-processing or pruning. |
| `best_CNN.h5`             | Best performing CNN model during training.                                 |
| `final_CNN.h5`            | Final CNN model used in evaluation and comparison.                         |
| `KNN_model.pkl`           | Pickled K-Nearest Neighbors model trained on extracted features.           |
| `KNN_scaler.pkl`          | Scaler used to normalize features before feeding into the KNN model.       |
| `optimized_KNN_model.pkl` | Optimized version of the KNN model with best hyperparameters.              |
| `knn_model_package.pkl`   | Full KNN pipeline including model, preprocessing steps, and metadata.      |
| `SVM-RBF_model.pkl`       | Support Vector Machine with RBF kernel.                                    |
| `svm_rbf_trained.h5`      | Saved SVM model in HDF5 format (alternative to `.pkl`).                    |
| `svm_model_package.pkl`   | Full SVM pipeline including model and preprocessing.                       |
| `model_metadata.json`     | Metadata file containing model details, parameters, and configuration.     |

### 🔒 Notes

- Models are saved in various formats depending on the framework:
  - `.h5` for Keras-based models (CNNs and AFCNN).
  - `.pkl` for Scikit-learn models (KNN and SVM).
- For inference or further evaluation, make sure to load the correct preprocessor (e.g., scalers) along with the model.
- Metadata in `model_metadata.json` can help reproduce the model setup or assist in deployment.


