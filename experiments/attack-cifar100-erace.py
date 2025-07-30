import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import SGD, Adam
import torchvision
import torchvision.transforms as transforms
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

from models import ResNet32

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

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408), 
            (0.2675, 0.2565, 0.2761) 
        )
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408), 
            (0.2675, 0.2565, 0.2761) 
        )
    ])
    
    benchmark = avl.benchmarks.SplitCIFAR100(
        n_experiences=10,
        return_task_id=False,
        fixed_class_order=list(range(100)),
        seed=args.seed,
        train_transform=train_transform,
        eval_transform=test_transform,
    )
    
    
    configs = yaml.load(open(f"configs/{args.dataset}/{args.method}.yaml", "r"), Loader=yaml.FullLoader)
    configs = create_default_args(configs, args)
    
    models = []
    save_dir = f'./pretrained_models/{args.dataset}/{args.method}'
    
    for idx in range(len(benchmark.test_stream)):
        model_dir = os.path.join(save_dir, f'{idx+1}.pth')
        model = avl.models.SlimResNet18(
            benchmark.n_classes
        ).to(device)
        load_model(model_dir, model, device, idx)
        models.append(model)
            
    print(f"Loaded {len(models)} models")
    
    print(f"Strating Evaluation....")
    
    total_samples = [0] * len(benchmark.test_stream)
    sample_counts = [0] * len(benchmark.test_stream)
    original_correct = [0] * len(benchmark.test_stream)
    fgsm_correct = [0] * len(benchmark.test_stream)
    pgd_correct = [0] * len(benchmark.test_stream)
    auto_correct = [0] * len(benchmark.test_stream)

    epsilon = 8/255
    num_steps = 10

    for attacker_model_idx, exp in enumerate(benchmark.test_stream):
        print(f"============ Attacking {attacker_model_idx+1}th experience =============")
        print(f"Start of experience: {exp.current_experience}")

        attacker_model = models[attacker_model_idx]
        attacker_model.eval()
        
        # A) Trasnfer to Latest Model
        target_model_idx = len(benchmark.test_stream)-1
        target_model = models[target_model_idx]
        target_model.eval()

        fgsm_attack = torchattacks.FGSM(attacker_model, eps=epsilon)
        pgd_attack = torchattacks.PGD(attacker_model, eps=epsilon, alpha=epsilon/num_steps, steps=num_steps, random_start=True)
        auto_attack = torchattacks.AutoAttack(attacker_model, eps=epsilon, n_classes=benchmark.n_classes, version='standard', verbose=False)        
        
        for dataset_idx, exp2 in enumerate(benchmark.test_stream):
            if dataset_idx > attacker_model_idx:
                break

            dataset = exp2.dataset
            batch_size = 128
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
            
            sample_counts[attacker_model_idx] = len(dataset)
            total_samples[attacker_model_idx] += len(dataset)

            for images, labels, _ in tqdm(dataloader, desc=f"Attacking on exp {target_model_idx}"):
                images = images.to(device).float()
                labels = labels.to(device)

                fgsm_images = fgsm_attack(images, labels)
                pgd_images = pgd_attack(images, labels)
                auto_images = auto_attack(images, labels)   
                
                with torch.no_grad():
                    outputs = target_model(images)
                    original_preds = outputs.argmax(dim=1)
                    original_correct[attacker_model_idx] += (original_preds == labels).sum().item()

                    correct_mask = (original_preds == labels)
                    if correct_mask.sum() > 0:
                        correct_images = images[correct_mask]
                        correct_labels = labels[correct_mask]
                        
                        correct_fgsm = fgsm_images[correct_mask] 
                        fgsm_outputs = target_model(correct_fgsm)
                        fgsm_preds = fgsm_outputs.argmax(dim=1)
                        fgsm_correct[attacker_model_idx] += (fgsm_preds != correct_labels).sum().item()
                        
                        correct_pgd = pgd_images[correct_mask]
                        pgd_outputs = target_model(correct_pgd)
                        pgd_preds = pgd_outputs.argmax(dim=1)
                        pgd_correct[attacker_model_idx] += (pgd_preds != correct_labels).sum().item()
                        
                        correct_auto = auto_images[correct_mask]
                        auto_outputs = target_model(correct_auto)
                        auto_preds = auto_outputs.argmax(dim=1)
                        auto_correct[attacker_model_idx] += (auto_preds != correct_labels).sum().item()

                
    print(f"Original ACC per models: ")
    for i in range(len(benchmark.test_stream)):
        print(f"\tModel {i+1}: {original_correct[i]}/{total_samples[i]} = {original_correct[i]/total_samples[i]:.4f}")
    print("\n--------------------------------")

    print(f"FGSM ASR per models: ")
    for i in range(len(benchmark.test_stream)):
        j=4
        print(f"\tASR from {i+1}th to {j+1}th model: {fgsm_correct[i]}/{original_correct[i]} = {fgsm_correct[i]/original_correct[i]:.4f}")
    print("\n--------------------------------")

    print(f"PGD ASR per models: ")
    for i in range(len(benchmark.test_stream)):
        j=4
        print(f"\tASR from {i+1}th to {j+1}th model: {pgd_correct[i]}/{original_correct[i]} = {pgd_correct[i]/original_correct[i]:.4f}")
    print("\n--------------------------------")

    print(f"AA ASR per models: ")
    for i in range(len(benchmark.test_stream)):
        j=4
        print(f"\tASR from {i+1}th to {j+1}th model: {auto_correct[i]}/{original_correct[i]} = {auto_correct[i]/original_correct[i]:.4f}")
    print("\n--------------------------------")
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed value")
    parser.add_argument("--dataset", type=str, default='cifar100')
    parser.add_argument("--method", type=str, default='erace')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    set_seed(args.seed)

    main(args)