# 🐦 Birdify - Bird Sound Classification

## 🌟 Introduction

**Birdify** is a machine learning project for classifying bird species through their sounds.  
This guide will help you set up the development environment in just a few steps.

---

## 🛠️ Setup Guide

### 📋 Prerequisites

- Python **3.8+**
- **Git**
- Recommended IDE: **VS Code** or **PyCharm**

---

### 🚀 Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/XavierRiera/Birdify.git
cd Birdify
```

#### 2. Set Up Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3.9 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Verify Installation

```bash
pip list
```

#### 5. Jupyter Notebook Setup (Optional)

```bash
pip install ipykernel
python -m ipykernel install --user --name=birdify-venv
```

---

## 🔄 Environment Management

To deactivate the virtual environment:

```bash
deactivate
```

To reactivate:

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

---

## 💡 Tips

- ✅ Always activate your virtual environment before working on the project  
- 🔄 Regularly update your dependencies  
- 📦 After installing new packages, run:
  ```bash
  pip freeze > requirements.txt
  ```
## 👥  Contributors
- Maheen Asad
- Cinta Carot
- Arnau Martín
- Zat Pros
- Silvia Riaño
- Xavier Riera
- Lluc Sayols

## 📄  License
The Birdify repository is released under the GNU General Public License v3.0 (GPL-3.0). This license grants users the freedom to run, study, share, and modify the software as long as derivative works also remain open and under the same license. The choice of GPL-3.0 aligns with the values of transparency, accessibility, and collaboration promoted by our team. It ensures that Birdify and any future improvements made by the community remain freely available and benefit the broader ecosystem of open-source bioacoustic tools.
