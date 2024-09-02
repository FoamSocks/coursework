#!/bin/bash

#SBATCH --job-name=data_rd
#SBATCH --nodes=1
##SBATCH --partition=barton
#SBATCH --time=3-00:00:00 
#SBATCH --mem=500G
#SBATCH --cpus-per-task=64
#SBATCH --output=_output/%j_data_read.txt 
#SBATCH --mail-type=END              
#SBATCH --mail-user=alexander.huang@nps.edu

. /etc/profile

module load lang/miniconda3/23.1.0
source activate thesis

date

# set can be test or train

echo "read data:" $1
# set, date
python data_read.py $1
