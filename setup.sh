mamba env create -f environment.yml
mamba activate demansia
mamba env config vars set CUDA_HOME=$CONDA_PREFIX
mamba activate demansia
git clone --branch v1.1.3.post1 https://github.com/Dao-AILab/causal-conv1d
pip install -e causal-conv1d/
pip install -e custom-mamba/
