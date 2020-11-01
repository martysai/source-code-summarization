#!/bin/bash
#SBATCH --job-name=tuned_training
#SBATCH --error=tuned_training-%j.err
#SBATCH --output=tuned_training-%j.log
#SBATCH --time=18:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

bash tt_code_train.sh
