# FairGU: Fairness-aware Graph Unlearning in Social Network

## Requirements

```bash
conda create --name fairgu python=3.8.10
conda activate fairgu

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
TORCH="1.13.0"
CUDA="cu117"
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

Run `pip install -r requirements.txt` to install remaining packages.

## How to Run

Run FairGU unlearning experiment:
```bash
python main.py --exp Unlearn
```

Run membership attack experiment:
```bash
python main.py --exp Attack
```
## Cite
```
@INPROCEEDINGS{fairgu2026luo,
  author={Renqiang Luo, Yongshuai Yang, Huafei Huang, Qing Qing, Mingliang Hou, Ziqi Xu, Yi Yu, Jingjing Zhou and Feng Xia},
  booktitle={Proceedings of the ACM Web Conference (WWW)}, 
  title={FairGU: Fairness-aware Graph Unlearning in Social Networks}, 
  year={2026},
}
```