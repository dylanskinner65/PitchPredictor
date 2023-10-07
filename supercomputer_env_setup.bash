#!/bin/bash

# Define environment name
env_name="pitchpredict"

# Define a list of packages with specific versions
conda_packages=("numpy=1.21.6" "matplotlib=3.5.2" "scikit-learn=0.23.2"
          "pandas=1.3.5" "pytorch=1.13.1")

pip_packages=("adabelief-pytorch=0.2.0" "tqdm=4.65.0")

# Activate the Conda environment
conda activate $env_name

# Install packages with specific versions using conda install
for package in "${conda_packages[@]}"; do
    conda install -n $env_name -y $package
done

# Install packages with specific versions using pip install
for package in "${pip_packages[@]}"; do
    pip install $package
done


# Deactivate the Conda environment
conda deactivate
