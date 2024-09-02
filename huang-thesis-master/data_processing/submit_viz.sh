#!/bin/bash

#SBATCH --job-name=viz
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00 
#SBATCH --mem=64G
#SBATCH --output=_output/viz_%j.txt 
#SBATCH --mail-type=END              
#SBATCH --mail-user=alexander.huang@nps.edu

. /etc/profile
#/share/spack/gcc-7.2.0/miniconda3-4.5.12-gkh/condabin/conda activate tf-2.6

module load lang/miniconda3/23.1.0
source activate thesis

date
echo "make viz output script, outbound test data 7/26 (outbound small rfc)"
python build_viz.py 