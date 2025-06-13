## 📊 Model Metrics

This directory contains the evaluation metrics and training history for all models trained in the **Birdify** project. These artifacts help analyze and compare model performance both visually and quantitatively.

The metrics are organized by model type: KNN, SVM, CNN, and AFCNN.

### 📁 Contents

| File                            | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| `confusion_matrix_KNN.png`      | Confusion matrix of the KNN classifier.                                    |
| `confusion_matrix_SVM.png`      | Confusion matrix of the SVM classifier.                                    |
| `confusion_matrix_CNN.png`      | Confusion matrix of the CNN model.                                         |
| `confusion_matrix_AFCNN.png`    | Confusion matrix of the AFCNN model.                                       |
| `evaluation_metrics_KNN.json`   | Precision, recall, F1-score, and accuracy for the KNN model.               |
| `evaluation_metrics_SVM.json`   | Precision, recall, F1-score, and accuracy for the SVM model.                                  |
| `evaluation_metrics_CNN.json`   | Precision, recall, F1-score, and accuracy for the CNN model.                                      |
| `evaluation_metrics_AFCNN.json` | Precision, recall, F1-score, and accuracy for the AFCNN model.                                         |
| `training_history_CNN.json`     | Training/validation loss and accuracy over epochs for CNN.                 |
| `training_history_CNN.png`      | Plot of training/validation metrics for CNN.                               |
| `training_history_AFCNN.json`   | Training/validation metrics for the AFCNN model.                           |
| `training_history-AFCNN.png`    | Plot of training/validation loss and accuracy for AFCNN.                   |

### 📌 Notes

- Confusion matrices are provided as `.png` images for quick visual inspection of classification results.
- Evaluation metrics are stored in `.json` format to allow easy parsing and integration in reports or dashboards.
- Training histories help monitor model convergence and identify overfitting or underfitting trends.

