#!/bin/bash

#SBATCH --job-name=Hout_10k
#SBATCH --output=_output/%j_H_out_gen_10k.txt
#SBATCH --partition=barton
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --nodes=1
#SBATCH --time=45-
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
SAMPLES=10000
echo "Direction: " $DIRECTION
echo "Number of Samples: " $SAMPLES

python hessian.py $DIRECTION $SAMPLES $SLURM_JOB_ID
echo "done"