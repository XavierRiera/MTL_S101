#!/bin/bash

# 🐦 Birdify - Bird Sound Classification Setup Script
# ---------------------------------------------------
# This script will set up your development environment for Birdify
# Includes both Windows (Git Bash) and macOS/Linux support

echo -e "\n\033[1;36m🌿 Starting Birdify Setup\033[0m\n"

# 1. Clone Repository
echo -e "\033[1;34m🔽 Cloning Repository...\033[0m"
git clone https://github.com/XavierRiera/Birdify.git
cd Birdify || exit

# 2. Create Virtual Environment
echo -e "\n\033[1;34m🛠️ Creating Virtual Environment...\033[0m"
if [[ "$OSTYPE" == "msys" ]]; then
    python -m venv .venv
    source .venv/Scripts/activate
else
    python3 -m venv .venv
    source .venv/bin/activate
fi

# 3. Install Dependencies
echo -e "\n\033[1;34m📦 Installing Dependencies...\033[0m"
pip install --upgrade pip
pip install -r requirements.txt

# 4. Optional Jupyter Setup
echo -e "\n\033[1;33m💡 Optional Jupyter Notebook Setup\033[0m"
read -p "Do you want to set up Jupyter Notebook? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install ipykernel
    python -m ipykernel install --user --name=birdify-venv
    echo -e "\033[1;32m✔ Jupyter kernel installed as 'birdify-venv'\033[0m"
fi

# Completion Message
echo -e "\n\033[1;32m🎉 Setup Completed Successfully!\033[0m"
echo -e "\n\033[1mNext Steps:\033[0m"
echo -e "1. To activate virtual environment later:"
if [[ "$OSTYPE" == "msys" ]]; then
    echo -e "   \033[1msource .venv/Scripts/activate\033[0m"
else
    echo -e "   \033[1msource .venv/bin/activate\033[0m"
fi
echo -e "2. To deactivate: \033[1mdeactivate\033[0m"
echo -e "\n\033[3mHappy bird watching! 🦜\033[0m\n"