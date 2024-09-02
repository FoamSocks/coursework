#!/bin/bash

#SBATCH --job-name=d_splt
#SBATCH --nodes=1
#SBATCH --time=12:00:00 
#SBATCH --mem=128G
##SBATCH --partition=barton
##SBATCH --gres=gpu:1
#SBATCH --output=_output/%j_data_split.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=alexander.huang@nps.edu

. /etc/profile

module load lang/miniconda3/23.1.0
source activate thesis

DIRECTION=$1
DATASET=$2

date
echo "-"
python data_split.py $1 $2
echo "done"