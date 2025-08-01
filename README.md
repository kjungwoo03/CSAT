# Exploring Cross-Stage Adversarial Transferability in Class-Incremental Continual Learning (MMSP 2025)

Official repository of CSAT

[Jungwoo Kim](https://kjungwoo03.github.io), [Jong-Seok Lee](https://mcml.yonsei.ac.kr/)  


## Summary
This paper investigates stage-transferred attacks in class-incremental continual learning (CSAT), showing that adversarial examples generated from earlier-stage models remain effective against later-stage models.

## Environmental Set-up
```bash
conda create -n csat python=3.10
conda activate csat
git clone https://github.com/kjungwoo03/CSAT.git
cd CSAT
pip install -r requirements.txt
```


## Quick Start
### 1. Train baseline model
The trained baseline models will be saved in the `./pretrained_models/{dataset}`.
```bash
python train.py --dataset cifar100 --method gdumb --seed 42
```

### 2. Stage-transferred attacks
```bash
python attack.py --dataset cifar100 --method gdumb --seed 42
```

### 3. Others
For additional evaluations (e.g. defense, model similarity, complexity, etc.), see `./utils` folder.


## How to cite
```
TBD.
```


