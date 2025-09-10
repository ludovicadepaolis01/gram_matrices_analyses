#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB             # memory requested, per cpu
#SBATCH --account=Sis25_piasini       # account name
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --job-name=par_optim
#SBATCH --mail-type=ALL
#SBATCH --output=/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/gen_%x.%A.%a.out
#SBATCH --error=/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/gen_%x.%A.%a.err
#SBATCH --array=0-14 #total number of models

module purge #unload any previously loaded modules to use a clean venv
module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/virtualenvs/dl/bin/activate

job_id_file="job_ids_optim.txt"
> "$job_id_file" #clear the job_ids.txt file at the start

MODELS=(vgg16 vgg19 alexnet resnet18 resnet34 resnet50 resnet101 resnet151 googlenet inceptionv3 squeezenet mobilenet densenet121 densenet161 densenet169 densenet201)
MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "Running model: $MODEL_NAME"

cd $HOME/lab/gram_matrices_analyses/analyses

BASE_OPTS="--partition=boost_usr_prod --account=Sis25_piasini -n 1 -c 8 --mem=32G -t 23:55:00 --gres=gpu:1"

wait_for_job() {
    local jid=$1
    sleep 5
    echo "$jid" >> "$job_id_file"
}

jid_pre=$(sbatch --parsable $BASE_OPTS \
    --job-name="${MODEL_NAME}_optimization" \
    --output="/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_gen_%A.out" \
    --error="/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_err_%A.err" \
    --wrap="python -u image_generation.py --model $MODEL_NAME")
wait_for_job "$jid_pre"

prev_jid=$jid_pre
for i in {1..2}; do
    jid=$(sbatch --parsable $BASE_OPTS \
        --dependency=afterany:$prev_jid \
        --job-name="${MODEL_NAME}_optimization" \
        --output="/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_gen_%A.out" \
        --error="/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_err_%A.err" \
        --wrap="python -u image_generation.py --model $MODEL_NAME")
    wait_for_job "$jid"
    prev_jid=$jid
done
