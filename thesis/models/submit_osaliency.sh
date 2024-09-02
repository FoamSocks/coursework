#!/bin/bash

#SBATCH --job-name=osal_gen
#SBATCH --output=_output/%j_osalgen.txt
#SBATCH --partition=barton
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=3-
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=END
#SBATCH --mail-user=alexander.huang@nps.edu

. /etc/profile

module load lang/miniconda3/23.1.0
source activate tf
nvidia-smi
hostname

date
echo "-"

DIRECTION=outbound
SAMPLES=all

echo "Direction: " $DIRECTION
echo "Num samples: " $SAMPLES

python saliencymap.py $DIRECTION $SAMPLES $SLURM_JOB_ID
echo "done"
