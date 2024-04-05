#!/usr/bin/env bash

#SBATCH --account=def-gerope
#SBATCH --time=0-00:20:00
#SBATCH --tasks=1
#SBATCH --output=output/%j.out
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G

lscpu
nvidia-smi

# Load modules for environment
module load StdEnv/2020
module load python/3.8.10
module load scipy-stack/2023a

# Create environment
#virtualenv --no-download ENV
source ENV/bin/activate

# update pip
#pip install --no-index --upgrade pip

# install reqs
#pip install -r requirements.txt

# copy data
#echo "Copying and extracting data"
#export DATASET_PATH=$SLURM_TMPDIR/GOT-10k
#mkdir -p $DATASET_PATH
#unzip -q ./data/GOT-10k.zip -d $DATASET_PATH

# run train
echo "Starting training"
python train.py

deactivate
