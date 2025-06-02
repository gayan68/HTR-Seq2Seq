#!/bin/bash
#SBATCH -A Berzelius-2025-71
#SBATCH -o /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.err
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -c 4                           # one CPU core
#SBATCH -t 2-00:00:00
#SBATCH --mem=40G

#Experiment ID
EXPERIMENT_ID="50"

# mamba init bash
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate pytorch25

# Parameters
file=main_torch_latest.py

root=/home/x_gapat/PROJECTS/codes/HTR-Seq2Seq
main_script="${root}/${file}"


echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="/home/x_gapat/PROJECTS/logs/HTR-Seq2Seq/${EXPERIMENT_ID}_${CURRENTDATE}_ID_${SLURM_JOB_ID}/"
echo "path log :"
echo ${PATHLOG}
mkdir -p ${PATHLOG}

output_file="${PATHLOG}/${SLURM_JOB_ID}.txt"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/HTR-Seq2Seq/


# The job
python -u $main_script \
	--dataset MIXED_100 \
	--run_id "$EXPERIMENT_ID" \
    --wandb 1 \

>> "$output_file" 2>&1
