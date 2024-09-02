#!/bin/bash

#SBATCH --job-name=dp_iot
#SBATCH --nodes=1
#SBATCH --time=24:00:00 
#SBATCH --mem=256G
##SBATCH --partition=barton
##SBATCH --gres=gpu:1
#SBATCH --output=data_proc_iot_%j.txt 
#SBATCH --mail-type=END              
#SBATCH --mail-user=alexander.huang@nps.edu

. /etc/profile
#/share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/condabin/conda activate tf-2.6

module load lang/miniconda3/4.5.12
source activate tf
nvidia-smi

python iot_data_processing.py 
