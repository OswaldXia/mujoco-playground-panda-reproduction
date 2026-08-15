#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
JUPYTER_BIN="${REPO_ROOT}/.venv/bin/jupyter"
RUNTIME_ROOT="${REPO_ROOT}/reproduction/artifacts/jupyter"
KERNEL_PREFIX="${RUNTIME_ROOT}/kernel-prefix"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${JUPYTER_BIN}" ]]; then
  echo "[FAIL] Project notebook environment is unavailable."
  echo "[NEXT] Create .venv and install the notebook extra from chapter 06:"
  echo "       python -m pip install -e '.[notebooks]'"
  exit 2
fi

mkdir -p "${RUNTIME_ROOT}/config" "${RUNTIME_ROOT}/ipython" \
  "${RUNTIME_ROOT}/matplotlib" "${KERNEL_PREFIX}"

"${PYTHON_BIN}" -m ipykernel install \
  --prefix "${KERNEL_PREFIX}" \
  --name python3 \
  --display-name "Panda Course (.venv)" >/dev/null

export JUPYTER_PATH="${KERNEL_PREFIX}/share/jupyter${JUPYTER_PATH:+:${JUPYTER_PATH}}"
export JUPYTER_CONFIG_DIR="${RUNTIME_ROOT}/config"
export IPYTHONDIR="${RUNTIME_ROOT}/ipython"
export MPLCONFIGDIR="${RUNTIME_ROOT}/matplotlib"

echo ""
echo "Panda course notebooks"
echo "========================================================================"
echo "  Repository     ${REPO_ROOT}"
echo "  Python         ${PYTHON_BIN}"
echo "  Kernel         Panda Course (.venv)"
echo "  Start page     docs/notebooks/README.md"
echo "------------------------------------------------------------------------"
echo "  Close Jupyter with Ctrl-C in this terminal."
echo ""

cd "${REPO_ROOT}"
exec "${JUPYTER_BIN}" lab docs/notebooks --notebook-dir="${REPO_ROOT}"
