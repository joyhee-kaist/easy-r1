# easy-r1

## Directory Change
```bash
export RAY_TMPDIR=/workspace/ray
export HF_HOME=/workspace/huggingface
export HF_DATASETS_CACHE=/workspace/hf_datasets
export TRANSFORMERS_CACHE=/workspace/transformers
```

## Conda
### Conda install
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh
source ~/.bashrc

```
### Conda env
```bash
conda create -n verl python==3.10.12
conda activate verl
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install --r requirements_docker.txt
```

### For runpod
```bash
apt-get update && apt-get install -y tmux
apt-get install -y htop
```
