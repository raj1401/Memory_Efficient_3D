#!/bin/bash

# Exit script on error
set -e

echo "Updating system and installing required dependencies..."
apt update && apt install -y wget git
rm -rf /var/lib/apt/lists/*

echo "Downloading and installing Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda
rm Miniconda3-latest-Linux-x86_64.sh

# Set up conda environment
export PATH="/opt/miniconda/bin:$PATH"
echo 'export PATH="/opt/miniconda/bin:$PATH"' >> ~/.bashrc

echo "Initializing Conda..."
conda init bash
conda config --set auto_activate_base false
conda create -y -n optimizing-gs python=3.10.0

# Activate conda environment and install required packages
echo "Installing required Python packages..."
source /opt/miniconda/bin/activate optimizing-gs

pip install numpy==1.26.3 scipy==1.14.1 scikit-learn==1.5.2 plyfile==1.1 PyYAML==6.0.2
conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt20cu118
pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.0.0_cu118.html
pip install streamlit==1.42.1 viser==0.2.3 nerfview==0.0.3

# Ensure Conda environment activates by default in new shell sessions
echo "conda activate optimizing-gs" >> ~/.bashrc

echo "Setup complete! Restart your terminal or run 'source ~/.bashrc' to apply changes."
