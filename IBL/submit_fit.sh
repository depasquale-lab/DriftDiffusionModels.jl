#!/bin/bash -l
# SGE array job for BU's Shared Computing Cluster (SCC): ONE TASK PER
# (subject, K, restart), so all restarts run in parallel and results populate
# incrementally as each restart finishes.
#
# Submit from the repository root:
#     qsub IBL/submit_fit.sh
#
# Subjects are the TOP-N by trial count and tasks are ordered subject-major, so
# the animals with the most trials are scheduled first. Layout with the defaults
# below (TOP=6, K in {3,4,5}, 10 restarts): 6 * 3 * 10 = 180 tasks, and the
# array must be submitted as -t 1-180 (task count = TOP * #K * RESTARTS —
# keep the -t line below in sync if you change TOP/KS/RESTARTS).
#
# Aggregate at any time (also mid-run) with:
#     julia --project=IBL IBL/collect_results.jl
#
# Before the first submission, instantiate the IBL environment once on the SCC:
#     julia --project=IBL -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate(); Pkg.precompile()'

#$ -N ibl_omission_ddm
#$ -P depaqlab
#$ -l h_rt=36:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
#$ -t 1-120
#$ -j y
#$ -o IBL/logs/$JOB_NAME.$JOB_ID.$TASK_ID.log

set -euo pipefail

# All four are overridable at submission without editing this file, e.g.
#     qsub -v KS="6",ITERATIONS=1000 -t 1-60 IBL/submit_fit.sh
# (keep the -t range = TOP * #KS * RESTARTS). A resubmission with a higher
# ITERATIONS warm-starts each already-finished restart from its saved model.
TOP=${TOP:-6}                     # number of subjects (most trials first)
read -ra KS <<< "${KS:-3 4 5}"    # DDM state counts (each fit adds one omission state)
RESTARTS=${RESTARTS:-10}          # random restarts per (subject, K)
ITERATIONS=${ITERATIONS:-1000}    # L-BFGS iteration target per restart

cd "${SGE_O_WORKDIR}"          # repository root (where qsub was run)
mkdir -p IBL/logs

# subject-major task layout: tasks 1..(NK*R) → subject 1, next block → subject 2, …
# within a subject: K-major, restart-minor (K1 r1..rR, K2 r1..rR, …)
NK=${#KS[@]}
PER_SUBJ=$(( NK * RESTARTS ))
SUBJ=$(( (SGE_TASK_ID - 1) / PER_SUBJ + 1 ))
REM=$(( (SGE_TASK_ID - 1) % PER_SUBJ ))
K=${KS[$(( REM / RESTARTS ))]}
RESTART=$(( REM % RESTARTS + 1 ))

export JULIA_NUM_THREADS=${NSLOTS:-1}
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/.julia}"

echo "host=$(hostname) task=${SGE_TASK_ID}/${SGE_TASK_LAST} subject_rank=${SUBJ}/${TOP} K=${K} restart=${RESTART} threads=${JULIA_NUM_THREADS}"

julia --project=IBL -t "${JULIA_NUM_THREADS}" IBL/fit_omission_hmm.jl \
    --top "${TOP}" \
    --task-id "${SUBJ}" \
    --K "${K}" \
    --restart-id "${RESTART}" \
    --iterations "${ITERATIONS}" \
    --out IBL/results
