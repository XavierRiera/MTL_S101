# 🐦 Birdify - Bird Sound Classification

## 📋 Project Structure
github-folder/
├── LICENSE
├── README.md
├── requirements.txt
└── src/
├── initial-setup.py
├── main.py
├── data/
│ ├── preprocessing/
│ └── feature_extraction/ # CSV outputs using Essentia
├── models/ # .py training files → outputs .h5
└── app/
├── app.py # Flask/FastAPI interface
└── templates/ # HTML/CSS
└── res/ # Bird images for interface


## 🛠️ Setup (Windows & macOS)

1. **Clone repo**  
```bash
git clone https://github.com/yourusername/birdify.git
cd birdify
Run initial setup
bash
python src/initial-setup.py
(Automatically:)

Creates virtual environment
Installs dependencies
Downloads dataset (via Kaggle API)
Activate environment
bash
# Windows
.\birdenv\Scripts\activate

# macOS
source birdenv/bin/activate
🔧 Usage

Train models

bash
python src/models/train_ast.py       # Best accuracy (Transformer)
python src/models/train_cnn_lstm.py  # Balanced approach
python src/models/train_xgboost.py   # Fastest (CPU-friendly)
Run interface

bash
python src/app/app.py
→ Access at http://localhost:5000

📊 Data Flow

MP3 files → src/data/preprocessing/
Feature extraction → CSV in src/data/feature_extraction/
Models save → .h5 files in src/models/
Interface loads .h5 + res/ images
👥 Team Collaboration

Windows users: Use Git Bash for CLI commands
macOS users: Ensure Python 3.8+ via Homebrew
Shared workflow:
Pull latest changes
Create feature branches
Test in virtual environment
Push to dev branch
⚠️ Troubleshooting

Kaggle API issues → Reinstall with pip install kaggle --upgrade
Essentia errors → Use prebuilt binaries for your OS
Missing dependencies → pip install -r requirements.txt
📧 Contact: team@birdify.edu
🔗 Dataset: Kaggle - 114 Bird Species
