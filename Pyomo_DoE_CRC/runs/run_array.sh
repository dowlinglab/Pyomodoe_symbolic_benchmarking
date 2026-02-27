#!/usr/bin/env bash
#SBATCH --job-name=pyomo_doe_bench
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-999%50

set -euo pipefail

module purge
module load python
module load gcc

export PATH="/Users/snarasi2/.idaes/bin:$PATH"
echo "which ipopt: $(which ipopt)"
ipopt -v

VENV_DIR="${PWD}/.venv"
TASKS_FILE="${PWD}/runs/tasks.tsv"
OUT_DIR="${PWD}/results/raw"
mkdir -p "${OUT_DIR}"

LINE="$(awk -v n=$((SLURM_ARRAY_TASK_ID+1)) 'NR==n {print; exit}' "${TASKS_FILE}")"
if [[ -z "${LINE}" ]]; then
  echo "No task line found for array id ${SLURM_ARRAY_TASK_ID}"
  exit 2
fi

PROBLEM="$(echo "${LINE}" | cut -f1)"
INSTANCE="$(echo "${LINE}" | cut -f2)"
MODE="$(echo "${LINE}" | cut -f3)"
SOLVER="$(echo "${LINE}" | cut -f4)"
RUN_ID="$(echo "${LINE}" | cut -f5)"
DERIV_CHECK="$(echo "${LINE}" | cut -f6)"
OBJECTIVE_OPTION="$(echo "${LINE}" | cut -f7)"
OUT_FILE="${OUT_DIR}/${RUN_ID}.json"

echo "Running: problem=${PROBLEM} instance=${INSTANCE} mode=${MODE} solver=${SOLVER} run_id=${RUN_ID} deriv_check=${DERIV_CHECK} objective_option=${OBJECTIVE_OPTION}"
"${VENV_DIR}/bin/python" -m bench.run \
  --problem "${PROBLEM}" \
  --instance "${INSTANCE}" \
  --mode "${MODE}" \
  --solver "${SOLVER}" \
  --out "${OUT_FILE}" \
  --deriv-check "${DERIV_CHECK}" \
  --objective-option "${OBJECTIVE_OPTION}"
