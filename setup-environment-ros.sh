#!/usr/bin/env bash
set -euo pipefail

# ---- settings ----
PY_VERSION="${PY_VERSION:-3.9.10}"   # target python (pyenv)
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> init pyenv (non-interactive shells)"
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv >/dev/null 2>&1; then
  eval "$(pyenv init -)"
else
  echo "ERROR: pyenv not found. Install pyenv first." >&2
  exit 1
fi

echo "==> ensure pyenv python ${PY_VERSION}"
pyenv install -s "${PY_VERSION}"

# absolute path to the desired python in pyenv
PYENV_PY="$("${PYENV_ROOT}/bin/pyenv" root)/versions/${PY_VERSION}/bin/python"
if [ ! -x "${PYENV_PY}" ]; then
  echo "ERROR: expected python not found: ${PYENV_PY}" >&2
  exit 1
fi

# optional: write .python-version
pyenv local "${PY_VERSION}" || true

echo "==> python to use"
"${PYENV_PY}" -V
echo "${PYENV_PY}"

VPY="${VENV_DIR}/bin/python"
VPIP="${VENV_DIR}/bin/pip"

echo "==> hard-isolate from system"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PIP_CONFIG_FILE || true

## for rospy
"${VPIP}" install pyyaml rospkg PyQt5 PySide2

echo "==> done."
echo "Activate with: source ${VENV_DIR}/bin/activate"
