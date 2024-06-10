#!/bin/bash
##SBATCH --mem=128G
#SBATCH --job-name=icl_experiment
#SBATCH -t 120:00:00  # time requested in hour:minute:second
#SBATCH --gres=gpu:2
#SBATCH --partition=compsci-gpu
#SBATCH --output=icl_out/%x-%j.out  # Save stdout to sout directory
#SBATCH --error=icl_out/%x-%j.err   # Save stderr to sout directory

python3 ./experiment.py base omniglot