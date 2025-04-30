🐦 Birdify - Bird Sound Classification
🌟 Introduction

Birdify is a machine learning project for classifying bird species through their sounds. This guide will help you set up the development environment.

🛠️ Setup Guide

📋 Prerequisites

Python 3.8+
Git
Recommended IDE: VS Code or PyCharm
🚀 Installation Steps

1. Clone the Repository

bash
git clone https://github.com/XavierRiera/Birdify.git
cd Birdify
2. Set Up Virtual Environment

Windows

bash
python -m venv .venv
.venv\Scripts\activate
macOS/Linux

bash
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies

bash
pip install --upgrade pip
pip install -r requirements.txt
4. Verify Installation

bash
pip list
5. Jupyter Notebook Setup (Optional)

bash
pip install ipykernel
python -m ipykernel install --user --name=birdify-venv
🔄 Environment Management

To deactivate virtual environment:

bash
deactivate
To reactivate:

Windows

bash
.venv\Scripts\activate
macOS/Linux

bash
source .venv/bin/activate
💡 Tips

Always activate your virtual environment before working on the project
Keep your dependencies updated regularly
Use pip freeze > requirements.txt to update requirements after adding new packages