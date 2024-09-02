#!/bin/bash

#SBATCH --job-name=orf
#SBATCH --output=_output/%j_orfc.txt
##SBATCH --partition=mccloud
##SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --time=10-
#SBATCH --cpus-per-task=64
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

DATA=outbound
MODEL=rfc

#DATA=$1
#MODEL=$2
#SAMPLE=$3

echo "Data: " $DATA
echo "Model: " $MODEL

python train.py $DATA $MODEL $SLURM_JOB_ID
echo "done"
