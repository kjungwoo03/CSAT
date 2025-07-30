import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import SGD, Adam
import torchvision
import torchvision.transforms as transform
import argparse
import avalanche as avl
import yaml
import os
from utils import set_seed, create_default_args
import torchattacks
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

def load_model(model_path, model, device, model_num):
    model_filename=model_path
    state_dict = torch.load(model_filename, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    
    print(f"Model {model_num+1} loaded from {model_filename}")
    
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(device)
    device_capability = torch.cuda.get_device_capability(device)
    device_mem = torch.cuda.get_device_properties(device).total_memory / 1024 **3

    print(f"Device name: {device_name}")
    print(f"Device capability: {device_capability}")
    print(f"Device memory: {device_mem:.2f}GB")

    mnist_transform = transform.Compose([
        transform.Normalize((0.1307,), (0.3081,))
    ])

    benchmark = avl.benchmarks.SplitMNIST(
        n_experiences=5,
        return_task_id=False,
        seed=args.seed,
        fixed_class_order=list(range(10)),
        train_transform=mnist_transform,
        eval_transform=mnist_transform,
    )
    
    configs = yaml.load(open(f"configs/{args.dataset}/{args.method}.yaml", "r"), Loader=yaml.FullLoader)
    configs = create_default_args(configs, args)
    
    models = []
    save_dir = f'./pretrained_models/{args.dataset}/{args.method}'
    
    for idx in range(len(benchmark.test_stream)):
        model_dir = os.path.join(save_dir, f'{idx+1}.pth')
        model = avl.models.SimpleMLP(
            num_classes=benchmark.n_classes,
            hidden_size = configs.hidden_size,
            hidden_layers = configs.hidden_layers,
            drop_rate = configs.dropout
        )
        load_model(model_dir, model, device, idx)
        models.append(model)
            
    print(f"Loaded {len(models)} models")
    
    print(f"Strating Evaluation....")
    
    total_samples = [0] * 2
    original_correct = [0] * 2
    pgd_correct = [0] * 2

    epsilon = 0.3
    num_steps = 40

    attacker_models = [3, 4]
    target_models = [4, 3]
    
    ai = -1
    for attacker_model_idx, target_model_idx in zip(attacker_models, target_models):
        ai += 1
        
        attacker_model = models[attacker_model_idx]
        attacker_model.eval()
        
        target_model = models[target_model_idx]
        target_model.eval()
        
        pgd_attack = torchattacks.PGD(attacker_model, eps=epsilon, alpha=epsilon/num_steps, steps=num_steps, random_start=True)
        
        for dataset_idx, exp2 in enumerate(benchmark.test_stream):
            if dataset_idx > min(attacker_model_idx, target_model_idx):
                continue
            
            dataset = exp2.dataset
            batch_size = 128
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
            
            total_samples[ai] += len(dataset)
            
            for images, labels, _ in tqdm(dataloader, desc=f"Attacking using model {ai}"):
                images = images.to(device).float()
                labels = labels.to(device)
                
                pgd_images = pgd_attack(images, labels)
                
                with torch.no_grad():
                    outputs = target_model(images)
                    original_preds = outputs.argmax(dim=1)
                    original_correct[ai] += (original_preds == labels).sum().item()
                    
                    correct_mask = (original_preds == labels)
                    if correct_mask.sum() > 0:
                        correct_images = images[correct_mask]
                        correct_labels = labels[correct_mask]
                        
                        correct_pgd = pgd_images[correct_mask]
                        pgd_outputs = target_model(correct_pgd)
                        pgd_preds = pgd_outputs.argmax(dim=1)
                        pgd_correct[ai] += (pgd_preds != correct_labels).sum().item()
                        
    print(f"Original ACC per models: ")
    print(f"Model 0: {original_correct[0]}/{total_samples[0]}={original_correct[0] / total_samples[0]:.4f}")
    print(f"Model 4: {original_correct[1]}/{total_samples[1]}={original_correct[1] / total_samples[1]:.4f}")
    
    print(f"PGD ACC per models: ")
    print(f"Model 0: {pgd_correct[0]}/{original_correct[0]}={pgd_correct[0] / original_correct[0]:.4f}")
    print(f"Model 4: {pgd_correct[1]}/{original_correct[1]}={pgd_correct[1] / original_correct[1]:.4f}")
        
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed value")
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--method", type=str, default='gdumb')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)