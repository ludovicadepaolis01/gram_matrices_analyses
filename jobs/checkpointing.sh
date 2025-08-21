#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1 #1 gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB             # memory requested, per cpu
#SBATCH --account=Sis25_piasini       # account name
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --job-name=checkpoints
#SBATCH --mail-type=ALL
##SBATCH --mail-user=ldepaoli@sissa.it

job_id_file="job_ids.txt"

# Clear the job_ids.txt file at the start
> "$job_id_file"

# Function to wait until the job is running or completed
#wait_for_job_running_or_completed() {
#    local jid=$1
#    while true; do
        # Check the state of the job using scontrol
#        job_state=$(scontrol show job "$jid" | grep "JobState=" | cut -d= -f2 | cut -d ' ' -f1)
#        if [ "$job_state" == "RUNNING" ] || [ "$job_state" == "COMPLETED" ]; then
#            echo "$jid" >> "$job_id_file"
#            break
#        elif [ "$job_state" == "FAILED" ] || [ "$job_state" == "CANCELLED" ]; then
#            echo "Job $jid failed or was cancelled." >> "$job_id_file"
#            break
#        fi
#        sleep 60  # Check every minute
#    done
#}

wait_for_job() {
    local jid=$1
    sleep 5
    echo "$jid" >> "$job_id_file"
}

jid_pre=$(sbatch --parsable \
    --export=MODEL_NAME="$MODEL_NAME" \
    --job-name="${MODEL_NAME}_optimization" \
    --output="/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_gen_%A.out" \
    --error="/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_err_%A.err" \
    optimization.sh)
wait_for_job "$jid_pre"

prev_jid=$jid_pre
for i in {1..3}; do
    jid=$(sbatch --parsable \
        --dependency=afterok:$prev_jid \
        --export=MODEL_NAME="$MODEL_NAME" \
        --job-name="${MODEL_NAME}_optimization" \
        --output="/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_gen_%A.out" \
        --error="/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/gen_out/${MODEL_NAME}_err_%A.err" \
        optimization.sh)
    wait_for_job "$jid"
    prev_jid=$jid
done
