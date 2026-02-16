#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --account=Sis25_piasini
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=par_optim_test
#SBATCH --mail-type=ALL
#SBATCH --output=/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/out/par_optim_test_%x.%A.%a.out
#SBATCH --error=/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/out/par_optim_test_%x.%A.%a.err
#SBATCH --array=0-64 #total number of models

module purge #unload any previously loaded modules to use a clean venv
module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/virtualenvs/dl/bin/activate

MODELS=(vgg16 vgg19 alexnet resnet18 resnet34 resnet50 resnet101 resnet152 inceptionv3 mobilenet densenet121 densenet169 densenet201)
TEXTURES=(blotchy matted pebble scaly striped)

N_TEXT=${#TEXTURES[@]}
MODEL_NAME="${MODELS[$((SLURM_ARRAY_TASK_ID / N_TEXT))]}"
TEXTURE_NAME="${TEXTURES[$((SLURM_ARRAY_TASK_ID % N_TEXT))]}"

cd "$HOME/lab/gram_matrices_analyses/analyses"
echo "Running model: $MODEL_NAME  texture: $TEXTURE_NAME"
python -u optimization_test.py --model "$MODEL_NAME" --texture "$TEXTURE_NAME"

