source /opt/conda/etc/profile.d/conda.sh
conda create --name filter python==3.10 -y

conda activate filter
pip install vllm -U