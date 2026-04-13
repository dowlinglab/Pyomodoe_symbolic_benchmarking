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
TASKS_FILE="${PWD}/runs/tasks_generated.tsv"

TASK_INDEX=$((SLURM_ARRAY_TASK_ID + 2))
LINE="$(awk -v n="${TASK_INDEX}" 'NR==n {print; exit}' "${TASKS_FILE}")"
if [[ -z "${LINE}" ]]; then
  echo "No task line found for array id ${SLURM_ARRAY_TASK_ID}"
  exit 2
fi

EXAMPLE_NAME="$(echo "${LINE}" | cut -f1)"
INSTANCE_NAME="$(echo "${LINE}" | cut -f2)"
INITIAL_POINT_NAME="$(echo "${LINE}" | cut -f3)"
OBJECTIVE_KEY="$(echo "${LINE}" | cut -f4)"
OBJECTIVE_OPTION="$(echo "${LINE}" | cut -f5)"
USE_GREY_BOX_OBJECTIVE="$(echo "${LINE}" | cut -f6)"
DERIVATIVE_MODE="$(echo "${LINE}" | cut -f7)"
GRADIENT_METHOD="$(echo "${LINE}" | cut -f8)"
RUN_REP="$(echo "${LINE}" | cut -f9)"
RUN_ID="$(echo "${LINE}" | cut -f10)"
OUT_JSON="$(echo "${LINE}" | cut -f11)"

RUN_DIR="$(dirname "${OUT_JSON}")/${RUN_ID}"
mkdir -p "$(dirname "${OUT_JSON}")" "${RUN_DIR}"

echo "Running: example=${EXAMPLE_NAME} instance=${INSTANCE_NAME} initial_point=${INITIAL_POINT_NAME} objective=${OBJECTIVE_KEY} derivative=${DERIVATIVE_MODE} run_id=${RUN_ID}"
"${VENV_DIR}/bin/python" -m benchmarking.runner \
  --example "${EXAMPLE_NAME}" \
  --instance "${INSTANCE_NAME}" \
  --initial-point "${INITIAL_POINT_NAME}" \
  --objective-key "${OBJECTIVE_KEY}" \
  --derivative-mode "${DERIVATIVE_MODE}" \
  --run-id "${RUN_ID}" \
  --out-json "${OUT_JSON}" \
  --run-dir "${RUN_DIR}"
