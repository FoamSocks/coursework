#!/bin/bash

#SBATCH --job-name=icd_inf
#SBATCH --nodes=1
#SBATCH --time=72:00:00 
#SBATCH --mem=128G
#SBATCH --partition=barton
#SBATCH --gres=gpu:a6000
#SBATCH --output=_output/%j_icd_inf.txt 
#SBATCH --mail-type=END              
#SBATCH --mail-user=alexander.huang@nps.edu

. /etc/profile

module load lang/miniconda3/23.1.0
source activate tf
nvidia-smi
hostname

date
echo "-"

# direction: inbound or outbound
DIRECTION=inbound
# model:
# concrete          concrete dropout (need a100)
# mcd               Monte Carlo dropout
# vi                variational inference
MODEL=concrete

echo "Direction: " $DIRECTION
echo "Model: " $MODEL

python bayesian_inference.py $DIRECTION $MODEL $SLURM_JOB_ID
echo "done"
