#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB             # memory requested, per cpu
#SBATCH --account=Sis25_piasini       # account name
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --job-name=crossval_checkpointing
#SBATCH --mail-type=ALL
#SBATCH --output=/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/gen_%x.%A.%a.out
#SBATCH --error=/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/gen_%x.%A.%a.err
#SBATCH --array=0-14 #total number of models

module purge #unload any previously loaded modules to use a clean venv
module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/virtualenvs/dl/bin/activate

MODELS=(vgg16 alexnet resnet18 resnet34 resnet50 resnet101 resnet151 googlenet inceptionv3 squeezenet mobilenet densenet121 densenet161 densenet169 densenet201)
MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "Running model: $MODEL_NAME"

cd $HOME/lab/gram_matrices_analyses/analyses

BASE_OPTS="--partition=boost_usr_prod --account=SIS25_piasini -n 1 -c 8 --mem=32G -t 23:55:00 --gres=gpu:1"

jid1=$(sbatch --parsable $BASE_OPTS \
  -J "chkpt_${MODEL_NAME}_1" \
  -o "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c1_%A.out" \
  -e "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c1_%A.err" \
  --wrap="python -u image_generation.py --model $MODEL_NAME")

jid2=$(sbatch --parsable --dependency=afterok:$jid1 $BASE_OPTS \
  -J "chkpt_${MODEL_NAME}_2" \
  -o "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c2_%A.out" \
  -e "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c2_%A.err" \
  --wrap="python -u image_generation.py --model $MODEL_NAME")

jid3=$(sbatch --parsable --dependency=afterok:$jid2 $BASE_OPTS \
  -J "chkpt_${MODEL_NAME}_3" \
  -o "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c3_%A.out" \
  -e "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c3_%A.err" \
  --wrap="python -u image_generation.py --model $MODEL_NAME")

jid4=$(sbatch --parsable --dependency=afterok:$jid3 $BASE_OPTS \
  -J "chkpt_${MODEL_NAME}_4" \
  -o "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c4_%A.out" \
  -e "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c4_%A.err" \
  --wrap="python -u image_generation.py --model $MODEL_NAME")

jid5=$(sbatch --parsable --dependency=afterok:$jid4 $BASE_OPTS \
  -J "chkpt_${MODEL_NAME}_5" \
  -o "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c5_%A.out" \
  -e "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c5_%A.err" \
  --wrap="python -u image_generation.py --model $MODEL_NAME")

jid6=$(sbatch --parsable --dependency=afterok:$jid5 $BASE_OPTS \
  -J "chkpt_${MODEL_NAME}_6" \
  -o "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c6_%A.out" \
  -e "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c6_%A.err" \
  --wrap="python -u image_generation.py --model $MODEL_NAME")

jid7=$(sbatch --parsable --dependency=afterok:$jid6 $BASE_OPTS \
  -J "chkpt_${MODEL_NAME}_7" \
  -o "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c7_%A.out" \
  -e "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c7_%A.err" \
  --wrap="python -u image_generation.py --model $MODEL_NAME")

jid8=$(sbatch --parsable --dependency=afterok:$jid7 $BASE_OPTS \
  -J "chkpt_${MODEL_NAME}_8" \
  -o "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c8_%A.out" \
  -e "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c8_%A.err" \
  --wrap="python -u image_generation.py --model $MODEL_NAME")

jid9=$(sbatch --parsable --dependency=afterok:$jid8 $BASE_OPTS \
  -J "chkpt_${MODEL_NAME}_9" \
  -o "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c9_%A.out" \
  -e "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c9_%A.err" \
  --wrap="python -u image_generation.py --model $MODEL_NAME")

jid10=$(sbatch --parsable --dependency=afterok:$jid9 $BASE_OPTS \
  -J "chkpt_${MODEL_NAME}_10" \
  -o "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c10_%A.out" \
  -e "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_c10_%A.err" \
  --wrap="python -u image_generation.py --model $MODEL_NAME")
