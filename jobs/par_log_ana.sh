#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --account=CMPNS_sissapia
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --job-name=par_log
#SBATCH --mail-type=ALL
#SBATCH --output=/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/out/par_log_%x.%A.%3a.out
#SBATCH --error=/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/out/par_log_%x.%A.%3a.err
#SBATCH --array=0-1 #total number of models

module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/virtualenvs/dl/bin/activate

#MODELS=(vgg16 vgg19 alexnet resnet18 resnet34 resnet50 resnet101 resnet152 inceptionv3 densenet121 densenet169 densenet201)
MODELS=(vgg19 vgg16)

MODEL_NAME="${MODELS[${SLURM_ARRAY_TASK_ID}]}"

cd "$HOME/lab/gram_matrices_analyses/analyses"
echo "Running model: $MODEL_NAME"

python -u log_analysis.py --model "$MODEL_NAME"
