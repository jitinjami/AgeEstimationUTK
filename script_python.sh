#!/bin/bash
#SBATCH --job-name=utk_crop_no_decay
#SBATCH --time=47:59:59
#SBATCH --output=messages/utk_crop_no_decay.out
#SBATCH --error=messages/utk_crop_no_decay.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -w icsnode11

USERNAME=$(id -un)
#---------------------------
#----------------------------
WORKING_DIR="/home/jami/age_UTK" # directory containing the script to be run
#---------------------------
# issue the job
#---------------------------

source ~/.bashrc
conda activate dex
cd $WORKING_DIR
# run the experiment
python3 train.py