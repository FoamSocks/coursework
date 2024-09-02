#!/bin/bash

#SBATCH --job-name=stinfer
#SBATCH --output=_output/%j_stinf.txt 
#SBATCH --nodes=1
#SBATCH --time=72:00:00 
#SBATCH --mem=96G
#SBATCH --partition=barton
#SBATCH --nodelist=compute-3-2
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END              
#SBATCH --mail-user=alexander.huang@nps.edu

. /etc/profile
#/share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/condabin/conda activate tf-2.6

module load lang/miniconda3/23.1.0
source activate tf
nvidia-smi
hostname

date
echo "-"

python st_inference.py $SLURM_JOB_ID
echo "done"
