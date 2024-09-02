#!/bin/bash
#SBATCH --partition=barton
#SBATCH --job-name=mobilenetv2
#SBATCH --nodes=1
##SBATCH --nodelist=compute-9-11
#SBATCH --gres=gpu:4
#SBATCH --time 1-
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

#SBATCH --mail-user=alexander.huang@nps.edu
#SBATCH --mail-type=ALL

#SBATCH --output=output/finetune/mobilenetv2_job_%j.txt

module load lang/miniconda3/23.1.0
source activate torch

MODEL=mobilenetv2

# mnet unfreeze: max=17, best=1
nvidia-smi
python ../trainer/task.py \
--model_dir="../models/"$MODEL"_"$SLURM_JOB_ID"_$(date +%Y-%m-%d_%H-%M-%S)/" \
--train_dir="/data/cs4321/HW1/train" \
--val_dir="/data/cs4321/HW1/validation" \
--test_dir="/data/cs4321/HW1/test" \
--model_type="$MODEL" \
--num_epochs=50 \
--batch_size=8 \
--num_classes=8 \
--fixed=False \
--unfreeze=8 \
--optimizer="sgd" \
--callback_list="checkpoint, csv_log" 



