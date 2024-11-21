

export WANDB_API_KEY="your key"


wandb login $WANDB_API_KEY

source ~/.bashrc

conda activate PyTorch


export TORCH_HOME="your path"

cd "your path"
python train.py
