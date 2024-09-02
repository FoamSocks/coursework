#!/bin/bash
#SBATCH --partition=barton
#SBATCH --job-name=vgg16
#SBATCH --nodes=1
##SBATCH --nodelist=compute-1-4
#SBATCH --gres=gpu:4
#SBATCH --time 1-
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

#SBATCH --mail-user=alexander.huang@nps.edu
#SBATCH --mail-type=ALL
# fixedfeature or finetune
#SBATCH --output=output/fixedfeature/vgg16_job_%j.txt

module load lang/miniconda3/23.1.0
source activate torch

MODEL=vgg16
# vgg unfreeze: max=5, best=2
nvidia-smi
python ../trainer/task.py \
--model_dir="../models/"$MODEL"_"$SLURM_JOB_ID"_$(date +%Y-%m-%d_%H-%M-%S)/" \
--train_dir="/data/cs4321/HW1/train" \
--val_dir="/data/cs4321/HW1/validation" \
--test_dir="/data/cs4321/HW1/test" \
--model_type="$MODEL" \
--num_epochs=40 \
--batch_size=8 \
--num_classes=8 \
--fixed=True \
--unfreeze=4 \
--optimizer="sgd" \
--callback_list="checkpoint, csv_log" 



