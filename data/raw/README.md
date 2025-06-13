## 🐦 Dataset

This directory contains the curated dataset used for training and evaluating the bird vocalization classifier in **Birdify**. The dataset has been cleaned and refined to ensure high-quality data for accurate model performance.


### 📄 `Birds_Voice.csv`

This CSV file contains metadata for each bird vocalization sample. It was compiled and curated from the [Xeno-Canto](https://xeno-canto.org/) bird sound archive, ensuring:
- Consistent formatting
- Sufficient audio length
- Valid and high-quality recordings
- Proper labeling and species verification

### 🧾 CSV Columns Explained

| Column            | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `common_name`     | Common name of the bird species (e.g., "Common Ostrich").                   |
| `scientific_name` | Scientific (Latin) name of the bird species, including subspecies if known. |
| `recordist_name`  | Name of the person who recorded the audio sample.                           |
| `recording_length`| Duration of the audio recording (in `mm:ss` format).                        |
| `Date`            | Date the recording was made (`YYYY-MM-DD`).                                 |
| `TYPE`            | Type of vocalization (e.g., "call", "song", "alarm").                       |
| `xc_id`           | Unique identifier for the recording on Xeno-Canto.                          |
| `Time`            | Time of day when the recording was captured.                                |
| `Country`         | Country where the recording took place.                                     |
| `Download_link`   | Direct URL to download the audio file from Xeno-Canto.                      |

### 📌 Notes

- This dataset was used as the basis for preprocessing, feature extraction, and model training in the [`notebooks/`](../notebooks) directory.
- Actual audio files corresponding to these records should be downloaded and stored separately using the `Download_link` column.


