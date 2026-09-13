#!/bin/bash -l
# SGE array job for BU's Shared Computing Cluster (SCC): one task per subject.
#
# Submit from the repository root:
#     qsub IBL/submit_fit.sh
#
# Each array task fits one (subject, K) pair: subjects are the top-N by trial
# count, N = (number of tasks) / (number of K values). With -t 1-12 and K in
# {1,2,3} that is the top 4 subjects. Adjust -P to your project.
# Before the first submission, instantiate the IBL environment once on the SCC:
#     module load julia
#     julia --project=IBL -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

#$ -N ibl_omission_ddm
#$ -P YOUR_PROJECT
#$ -l h_rt=48:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
#$ -t 1-12
#$ -j y
#$ -o IBL/logs/$JOB_NAME.$JOB_ID.$TASK_ID.log

set -euo pipefail

module load julia

cd "${SGE_O_WORKDIR}"          # repository root (where qsub was run)
mkdir -p IBL/logs

# One array task per (subject, K): tasks 1-3 → subject 1 with K=1,2,3, tasks 4-6 → subject 2, …
KS=(1 2 3)
NK=${#KS[@]}
TOP=$(( (SGE_TASK_LAST + NK - 1) / NK ))          # number of subjects = tasks / K values
SUBJ=$(( (SGE_TASK_ID - 1) / NK + 1 ))
K=${KS[$(( (SGE_TASK_ID - 1) % NK ))]}

export JULIA_NUM_THREADS=${NSLOTS:-1}
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/.julia}"

echo "host=$(hostname) task=${SGE_TASK_ID}/${SGE_TASK_LAST} subject_rank=${SUBJ}/${TOP} K=${K} threads=${JULIA_NUM_THREADS}"

# Timing guide (44k trials, 8 threads): ~35 s/iteration at K=1, ~70-80 s at K=2-3,
# so 2 restarts x 200 iterations at K=3 is roughly 9 h.
julia --project=IBL -t "${JULIA_NUM_THREADS}" IBL/fit_omission_hmm.jl     --top "${TOP}"     --task-id "${SUBJ}"     --K "${K}"     --restarts 2     --iterations 200     --out IBL/results
