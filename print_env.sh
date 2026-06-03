#!/usr/bin/env bash
# Copyright (c) 2023, NVIDIA CORPORATION.
# Reports relevant environment information useful for diagnosing and
# debugging GMMTS issues.
# Usage:
# "./print_env.sh" - prints to stdout
# "./print_env.sh > env.txt" - prints to file "env.txt"

print_env() {
echo "**git***"
if [ "$(git rev-parse --is-inside-work-tree 2>/dev/null)" == "true" ]; then
git log --decorate -n 1
echo "**git submodules***"
git submodule status --recursive
else
echo "Not inside a git repository"
fi
echo

echo "***OS Information***"
if [ -f /etc/os-release ]; then
  cat /etc/os-release
elif [ -f /etc/system-version ]; then
  cat /etc/system-version
fi
uname -a
echo

echo "***GPU Information***"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi
echo

echo "***Python***"
which python && python -c "import sys; print('Python {0}.{1}.{2}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available())" 2>/dev/null || echo "PyTorch not installed"
echo

echo "***Environment Variables***"
printf '%-32s: %s\n' MM_TSFLIB_PATH "${MM_TSFLIB_PATH:-<not set>}"
printf '%-32s: %s\n' CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES:-<not set>}"
printf '%-32s: %s\n' CONDA_PREFIX "${CONDA_PREFIX:-<not set>}"
echo

if type "conda" &> /dev/null; then
echo '***conda packages***'
which conda && conda list
echo
elif type "pip" &> /dev/null; then
echo "***pip packages***"
which pip && pip list
echo
else
echo "Neither conda nor pip found"
fi
}

echo "<details><summary>Click here to see environment details</summary><pre>"
echo "     "
print_env | while read -r line; do
    echo "     $line"
done
echo "</pre></details>"
