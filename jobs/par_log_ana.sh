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

module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/virtualenvs/dl/bin/activate

job_id_file="job_ids_log.txt"
> "$job_id_file" #clear the job_ids.txt file at the start

MODELS=(vgg19)
MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "Running model: $MODEL_NAME"

cd $HOME/lab/gram_matrices_analyses/analyses

BASE_OPTS="--partition=boost_usr_prod --account=CMPNS_sissapia -n 1 -c 8 --mem=128G -t 23:55:00 --gres=gpu:1" 

wait_for_job() {
    local jid=$1
    sleep 5
    echo "$jid" >> "$job_id_file"
}

jid_pre=$(sbatch --parsable $BASE_OPTS \
    --job-name="${MODEL_NAME}_par_log" \
    --output="/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/out/${MODEL_NAME}_log_%A.out" \
    --error="/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/out/${MODEL_NAME}_log_err_%A.err" \
    --wrap="python -u log_analysis.py --model $MODEL_NAME")
wait_for_job "$jid_pre"

prev_jid=$jid_pre
for i in {1..2}; do
    jid=$(sbatch --parsable $BASE_OPTS \
        --dependency=afterany:$prev_jid \
        --job-name="${MODEL_NAME}_par_log" \
        --output="/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/out/${MODEL_NAME}_log_%A.out" \
        --error="/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/out/${MODEL_NAME}_err_%A.err" \
        --wrap="python -u log_analysis.py --model $MODEL_NAME")
    wait_for_job "$jid"
    prev_jid=$jid
done
