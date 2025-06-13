## 📁 Data

This directory contains all data resources used in **Birdify**, structured into raw metadata and processed features ready for model training.

### 📂 `raw/` — Original Metadata


**File:** `Birds_Voice.csv`  
This file contains the original bird vocalization metadata. 

**File:** `birdcall_metadata_TOP50.csv`
This file contains the final curated version of the bird vocalization metadata. It was carefully cleaned and filtered to ensure high-quality samples and reliable labels for training.

#### 📄 CSV Columns

| Column            | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `common_name`     | Common name of the bird species (e.g., "Common Ostrich").                   |
| `scientific_name` | Scientific name, including subspecies if available.                         |
| `recordist_name`  | Name of the person who recorded the sample.                                 |
| `recording_length`| Duration of the audio clip (`mm:ss`).                                       |
| `Date`            | Recording date (`YYYY-MM-DD`).                                              |
| `TYPE`            | Type of vocalization (e.g., "call", "song", etc.).                          |
| `xc_id`           | Unique ID of the sample in the [Xeno-Canto](https://xeno-canto.org) archive.|
| `Time`            | Time of day when the recording was made.                                    |
| `Country`         | Country where the bird was recorded.                                        |
| `Download_link`   | Direct URL to download the audio file.                                      |

This file was used as the foundation for preprocessing and feature extraction.

### 📂 `features/` — Processed Feature Set

**File:** `birdcall_features_TOP50.csv`  
This file contains the final feature dataset used to train the best-performing models in Birdify. It includes feature vectors extracted from audio samples belonging to the **50 bird species** that yielded the highest classification accuracy.

This features were extracted from the `02_Feature_Extraction.ipynb` notebook and were used in the training models notebooks:
- `03_KNN.ipynb`, `04_SVM.ipynb`, `05_CNN.ipynb`, `06_AFCNN.ipynb`

### 📌 Notes

- This dataset was used as the basis for preprocessing, feature extraction, and model training in the [`notebooks/`](../notebooks) directory.
- Actual audio files corresponding to these records should be downloaded and stored separately using the `Download_link` column.


