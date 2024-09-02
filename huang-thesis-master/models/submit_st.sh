#!/bin/bash

#SBATCH --job-name=st(1,.7)
#SBATCH --output=_output/%j_st(1,.7).txt 
#SBATCH --nodes=1
#SBATCH --time=72:00:00 
#SBATCH --mem=96G
#SBATCH --cpus-per-task=64
#SBATCH --partition=barton
#SBATCH --gres=gpu:a6000
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

python student_teacher.py $SLURM_JOB_ID
echo "done"
