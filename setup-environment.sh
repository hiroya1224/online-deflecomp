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

echo "==> recreate venv: ${VENV_DIR} (using ${PY_ENV_PY:-$PYENV_PY})"
rm -rf "${VENV_DIR}"
"${PYENV_PY}" -m venv "${VENV_DIR}"

VPY="${VENV_DIR}/bin/python"
VPIP="${VENV_DIR}/bin/pip"

echo "==> hard-isolate from system"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PIP_CONFIG_FILE || true

echo "==> bootstrap pip into venv (no system pip)"
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
"${VPY}" /tmp/get-pip.py --no-warn-script-location
"${VPIP}" --version

echo "==> upgrade build basics"
"${VPIP}" install --upgrade pip setuptools wheel

echo "==> cleanup possibly conflicting pkgs"
"${VPIP}" uninstall -y numpy pinocchio pin scipy matplotlib scikit-robot || true
"${VPIP}" cache purge || true

echo "==> install deps in safe order (known-good combo)"
# 1) numpy (wheel only)
"${VPIP}" install --only-binary=:all: "numpy==1.26.4"
# 2) scipy, matplotlib
"${VPIP}" install "scipy==1.13.*" "matplotlib==3.9.*"
# 3) pinocchio (PyPI name: pin; import name: pinocchio)
"${VPIP}" install "pin==2.7.*"
# 4) scikit-robot (newer to match scipy>=1.13)
"${VPIP}" install "scikit-robot>=0.0.36"
# 5) ikpy==3.4.2
"${VPIP}" install "ikpy==3.4.2"


echo "==> compute site-packages dir (for guard)"
SP_DIR="$("${VPY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

echo "==> install sys.path guard (filter out system dist-packages)"
mkdir -p "${SP_DIR}"
cat > "${SP_DIR}/sitecustomize.py" <<'PY'
# Guard: remove system-wide dist-packages which can shadow venv wheels.
import sys
BAD = (
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3/dist-packages",
    "/usr/lib64/python3/dist-packages",
)
sys.path[:] = [p for p in sys.path if not any(p.startswith(b) for b in BAD)]
PY

echo "==> verify no system dist-packages is visible"
"${VPY}" - <<'PY'
import sys
bad="/usr/lib/python3/dist-packages"
print("python:", sys.executable)
for p in sys.path: print("  ", p)
assert not any(p.startswith(bad) for p in sys.path), f"found {bad} in sys.path"
PY

echo "==> sanity import test (use 'import pinocchio')"
"${VPY}" - <<'PY'
import sys, numpy, pinocchio, skrobot, scipy, matplotlib, ikpy
print("numpy :", numpy.__version__, numpy.__file__)
print("pino  :", pinocchio.__version__)
print("skrob :", skrobot.__version__)
print("scipy :", scipy.__version__)
print("mpl   :", matplotlib.__version__)
print("ikpy  :", ikpy.__version__)
print("env ok")
PY

echo "==> done."
echo "Activate with: source ${VENV_DIR}/bin/activate"
