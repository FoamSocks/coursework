#!/bin/bash

#SBATCH --job-name=idnn
#SBATCH --output=_output/%j_idnn.txt
#SBATCH --partition=barton
#SBATCH --gres=gpu:a6000
#SBATCH --mem=128G
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
# models.py arguments:
# data_option:  inbound OR outbound
# model:        log             logistic regression with l2, adam opt
#               log_vi          logisitic regression with variational inference
#               rfc             Random Forest Classifier
#               dnn             MLP with standard dropout
#               mcd             MLP with Monte Carlo dropout
#               nn_vi           MLP with variational inference layer
# sample        rus             Random Undersampling
#               ros             Random Oversampling
#               smote           SMOTE
#               no_sample       No sampling

DATA=inbound
MODEL=dnn

#DATA=$1
#MODEL=$2
#SAMPLE=$3

echo "Data: " $DATA
echo "Model: " $MODEL

python train.py $DATA $MODEL $SLURM_JOB_ID
echo "done"
