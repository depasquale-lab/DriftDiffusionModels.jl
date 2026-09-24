#!/bin/bash -l
# Aggregate finished restarts into shareable CSVs. Submit held on the fit array
# so it runs automatically once every restart is done:
#     qsub -hold_jid ibl_omission_ddm IBL/collect.sh
# or run it directly at any time for a snapshot of what has finished so far:
#     julia --project=IBL IBL/collect_results.jl

#$ -N ibl_omission_collect
#$ -P depaqlab
#$ -l h_rt=04:00:00
#$ -l mem_per_core=8G
#$ -pe omp 4
#$ -j y
#$ -o IBL/logs/$JOB_NAME.$JOB_ID.log

set -euo pipefail
cd "${SGE_O_WORKDIR}"
mkdir -p IBL/logs
export JULIA_NUM_THREADS=${NSLOTS:-1}
julia --project=IBL -t "${JULIA_NUM_THREADS}" IBL/collect_results.jl
