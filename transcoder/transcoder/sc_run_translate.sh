#!/bin/bash
#SBATCH --job-name=TC-Translate
#SBATCH --error=TC-Translate-%j.err
#SBATCH --output=TC-Translate-%j.log
#SBATCH --time=10:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

python translate.py --src_lang cpp --tgt_lang python --model_path models/model_2.pth < input_code.cpp

