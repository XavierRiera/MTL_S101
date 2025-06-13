## 📓 Notebooks

This directory contains the main Jupyter notebooks used throughout the development of **Birdify**, a bird species classifier based on their vocalizations using Machine Learning and Deep Learning techniques.

Each notebook corresponds to a specific stage in the pipeline, from data exploration to model training and evaluation.

### 🗂️ Notebook Contents

| File                         | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `00_Dataset_Analysis.ipynb`  | Initial exploration of the dataset. Visualizes species distribution and analyzes data quality. |
| `01_Data_Preprocessing.ipynb`| Dataset preparation: cleaning, relabeling, and normalizing audio samples.   |
| `02_Feature_Extraction.ipynb`| Acoustic feature extraction such as MFCCs, spectrograms, etc.               |
| `03_KNN.ipynb`               | Training and evaluation of a K-Nearest Neighbors classifier.               |
| `04_SVM.ipynb`               | Training and testing a Support Vector Machine for species classification.  |
| `05_CNN.ipynb`               | First deep learning approach using Convolutional Neural Networks.          |
| `06_AFCNN.ipynb`             | Final model using an Attention-based Fully Convolutional Neural Network.   |
| `info_species.csv`           | Auxiliary file with metadata about the bird species used.                  |

### ℹ️ Additional Notes

- The notebooks are intended to be run in sequence, from `00_` to `02_`, then the machine learning models can be executed independently.
- Make sure all dependencies are installed and the directory structure is in place before running the notebooks.
