mamba env create -f environment.yml
mamba activate demansia
mamba install cuda-toolkit -c nvidia -y # when install with environment.yml, cuda sometimes explodes.
git clone --branch v1.1.3.post1 https://github.com/Dao-AILab/causal-conv1d
pip install -e causal-conv1d/
pip install -e custom-mamba/
