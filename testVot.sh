#!/usr/bin/env bash

#SBATCH --account=def-gerope
#SBATCH --time=0-02:00:00
#SBATCH --tasks=1
#SBATCH --output=output/%j.out
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

lscpu
nvidia-smi

# Load modules for environment
module load StdEnv/2020
module load python/3.8.10
module load scipy-stack/2023a

# Create environment
virtualenv --no-download ENV
source ENV/bin/activate

# update pip
pip install --no-index --upgrade pip

# install reqs
pip install -r requirements.txt

# run train
echo "Starting testing"
python testVot.py

deactivate
