#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:00:10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB             # memory requested, per cpu
#SBATCH --account=Sis25_piasini       # account name
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --job-name=correlations
#SBATCH --mail-type=ALL
##SBATCH --mail-user=ldepaoli@sissa.it
#SBATCH --output=/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/out/corr_%x.%A.%3a.out
#SBATCH --error=/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/out/corr_%x.%A.%3a.err

module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/virtualenvs/dl/bin/activate

cd $HOME/lab/gram_matrices_analyses/analyses

python brainscore_correlations.py
