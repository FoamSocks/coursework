#!/bin/bash

#SBATCH --job-name=imcd
#SBATCH --nodes=1
#SBATCH --time=72:00:00 
#SBATCH --mem=96G
#SBATCH --partition=barton
#SBATCH --gres=gpu:a6000
#SBATCH --output=_output/%j_mcd.txt 
#SBATCH --mail-type=END              
#SBATCH --mail-user=alexander.huang@nps.edu

. /etc/profile
#/share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/condabin/conda activate tf-2.6

module load lang/miniconda3/23.1.0
source activate tf
# use a6000 or a100
nvidia-smi
hostname
date
echo "-"

DATA=inbound
MODEL=mcd

echo "Direction: " $DATA
echo "Model: " $MODEL

python train.py $DATA $MODEL $SLURM_JOB_ID

echo "done"
