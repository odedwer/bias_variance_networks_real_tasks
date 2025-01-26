#!/bin/bash
#SBATCH --time=32:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=~/lab/logs/alexnet_main_%j.log
#SBATCH --error=~/lab/logs/alexnet_main_error_%j.log

source ~/lab/venvs/idr_cnn/bin/activate
sbatch python3 ~/lab/idr_cnn/alexnet_train.py