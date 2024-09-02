#!/bin/bash

#SBATCH --job-name=ovi_inf
#SBATCH --output=_output/%j_ovi_inf.txt 
#SBATCH --nodes=1
#SBATCH --time=72:00:00 
#SBATCH --mem=96G
#SBATCH --partition=monaco
#SBATCH --gres=gpu:1
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
DIRECTION=outbound
# model:
# concrete          concrete dropout
# mcd               Monte Carlo dropout
# vi                variational inference
MODEL=vi

echo "Direction: " $DIRECTION
echo "Model: " $MODEL

python bayesian_inference.py $DIRECTION $MODEL $SLURM_JOB_ID
echo "done"
