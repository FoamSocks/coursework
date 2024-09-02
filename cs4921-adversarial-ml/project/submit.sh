#!/bin/bash

#SBATCH --job-name=ohe_atk
#SBATCH --output=_output/%j_ohe_atk.txt
#SBATCH --partition=mccloud
#SBATCH --gres=gpu:a6000
#SBATCH --mem=192G
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

# geo or port
OPTION=$1
echo "option: $1"

python attack.py $OPTION 

echo "done"
