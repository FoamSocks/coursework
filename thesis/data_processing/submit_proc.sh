#!/bin/bash

#SBATCH --job-name=it_proc
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00 
#SBATCH --cpus-per-task=6
#SBATCH --mem=192G
#SBATCH --output=_output/%j_data_proc.txt 
#SBATCH --mail-type=END              
#SBATCH --mail-user=alexander.huang@nps.edu

. /etc/profile

module load lang/miniconda3/23.1.0
source activate thesis

# inbound or outbound
DIRECTION=$1
# test or train
DATASET=$2

date
echo "Data Processing Direction:" $DIRECTION
echo "Data Processing Set:" $DATASET
python data_processing.py $DIRECTION $DATASET
