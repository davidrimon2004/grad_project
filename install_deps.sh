#!/usr/bin/env bash
# Installs the missing dependencies for the grad project.
# Run as user aitech, no sudo. To run it in the background and keep it going
# after you close the terminal:
#
#   nohup bash install_deps.sh > ~/install_log.txt 2>&1 &
#
# Check progress:   tail -5 ~/install_log.txt
# Finished when the last line is:  INSTALL_OK

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate videollava
python --version

cd ~/hdd/grad_project_david/grad_project_david
grep -q "torch==2.1.2" requirements.txt || {
    echo "ERROR: requirements.txt in this folder is not the pinned version."
    exit 1
}

pip install -r requirements.txt

conda install -y -c nvidia/label/cuda-12.1.0 cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX
nvcc --version

pip install -e ".[train]"
pip install -r requirements.txt

pip install flash-attn==2.5.9.post1 --no-build-isolation
pip install "git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d"

pip check || echo "(pip check reported issues above)"
python -c "import deepspeed, flash_attn, decord, transformers, peft, h5py; print('OK')"
echo INSTALL_OK