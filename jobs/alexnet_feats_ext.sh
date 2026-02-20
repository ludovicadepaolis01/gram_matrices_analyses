#!/bin/bash
#SBATCH --account=Sis25_piasini
#SBATCH --partition=boost_usr_prod
#SBATCH --time=23:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --job-name=alex_fe
#SBATCH --mail-type=ALL
#SBATCH --output=/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/out/alex_fe_%x.%A.%3a.out
#SBATCH --error=/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/out/alex_fe_%x.%A.%3a.err
#SBATCH --array=0-4 #n indices

module purge #unload any previously loaded modules to use a clean venv
module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/virtualenvs/dl/bin/activate

INDICES=(0 3 6 8 10)

IDX="${INDICES[$SLURM_ARRAY_TASK_ID]}"
echo "Running layer: $IDX"

cd /leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/analyses/

python -u alexnet_feature_extraction.py --index "$IDX"
