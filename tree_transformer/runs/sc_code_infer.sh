#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --error=inference-%j.err
#SBATCH --output=inference-%j.log
#SBATCH --time=2:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

bash tt_code_infer.sh
