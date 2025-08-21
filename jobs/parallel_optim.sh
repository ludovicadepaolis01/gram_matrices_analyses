#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --ntasks=1 #controls MPI tasks 
#SBATCH --gpus-per-task=1 #1 gpu
#SBATCH --cpus-per-task=8 #8 cpu cores per gpu
#SBATCH --mem=32GB #memory requested, per cpu. 512 is max, always start small
#SBATCH --account=Sis25_piasini #account name
#SBATCH --partition=boost_usr_prod #partition name
#SBATCH --job-name=parallel_optim
#SBATCH --array=0-14 #it should be the n. of models in __init__.py
#SBATCH --mail-type=ALL
##SBATCH --mail-user=ldepaoli@sissa.it
#SBATCH --output=/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/parallel_gen_%x.%A.%3a.out
#SBATCH --error=/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/parallel_gen_%x.%A.%3a.err

module purge #unload any previously loaded modules to use a clean venv
module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/virtualenvs/dl/bin/activate

cd $HOME/lab/gram_matrices_analyses/models

MODELS=(vgg16 alexnet resnet18 resnet34 resnet50 resnet101 resnet151 googlenet inceptionv3 squeezenet mobilenet densenet121 densenet161 densenet169 densenet201)
MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "Running model: $MODEL_NAME"

python --unbuffered image_generation.py 
