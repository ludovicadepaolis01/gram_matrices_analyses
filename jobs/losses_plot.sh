#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB             # memory requested, per cpu
#SBATCH --account=Sis25_piasini       # account name
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --job-name=vgg_losses
#SBATCH --mail-type=ALL
##SBATCH --mail-user=ldepaoli@sissa.it
#SBATCH --output=/leonardo/home/userexternal/ldepaoli/lab/gram_project/vgg_analyses/vgg_gen_out/losses_plots_%x.%A.%3a.out
#SBATCH --error=/leonardo/home/userexternal/ldepaoli/lab/gram_project/vgg_analyses/vgg_gen_out/losses_plots_%x.%A.%3a.err

module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/virtualenvs/dl/bin/activate

cd $HOME/lab/gram_project/vgg_analyses/src

srun --unbuffered python losses_plots.py
