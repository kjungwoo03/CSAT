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
from torch.nn.utils import parameters_to_vector
from itertools import combinations
from models import FeatureExtractorMLP
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import MNIST

def icarl_mnist_augment_data(img):
    img = img.numpy()
    padded = np.pad(img, ((0, 0), (2, 2), (2, 2)), mode='constant')
    random_cropped = np.zeros(img.shape, dtype=np.float32)
    crop = np.random.randint(0, high=4 + 1, size=(2,))

    if np.random.randint(2) > 0:
        random_cropped[:, :, :] = \
            padded[:, crop[0]:(crop[0]+28), crop[1]:(crop[1]+28)]
    else:
        random_cropped[:, :, :] = \
            padded[:, crop[0]:(crop[0]+28), crop[1]:(crop[1]+28)][:, :, ::-1]
    t = torch.tensor(random_cropped)

def icarl_aug(img):
    img = img.numpy()
    pad = np.pad(img, ((0,0),(2,2),(2,2)))
    i, j = np.random.randint(0,5,2)
    patch = pad[:, i:i+28, j:j+28]
    if np.random.rand() < .5:
        patch = patch[:, :, ::-1]
    return torch.tensor(patch, dtype=torch.float)

train_tf = T.Compose([T.Lambda(icarl_aug),
                      T.Normalize((0.1307,), (0.3081,))])
test_tf  = T.Compose([T.ToTensor(),
                      T.Normalize((0.1307,), (0.3081,))])

def disable_relu_inplace(m):
    for n, mod in m.named_children():
        if isinstance(mod, nn.ReLU) and mod.inplace:
            setattr(m, n, nn.ReLU(False))
        else:
            disable_relu_inplace(mod)

def load_state(mdl, pth, dev):
    mdl.load_state_dict(torch.load(pth, map_location=dev))
    disable_relu_inplace(mdl)
    return mdl.to(dev).eval()

def load_model(model_path, model, device, model_num):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    print(f"Model {model_num+1} loaded from {model_path}")

def get_models(args, configs,benchmark, device):
    models = []
    save_dir = f'./pretrained_models/{args.dataset}/{args.method}'
    
    if args.dataset == 'mnist':
        if args.method == 'icarl':
            for idx in range(len(benchmark.test_stream)):
                model_dir = os.path.join(save_dir, f'{idx+1}.pth')
                model = avl.models.make_icarl_net(
                    num_classes=benchmark.n_classes,
                    n=2,
                    c=1
                ).apply(avl.models.initialize_icarl_net)
                load_model(model_dir, model, device, idx)
                models.append(model)
        elif args.method == 'gdumb' or args.method == 'replay':
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
        elif args.method == 'erace':
            for idx in range(len(benchmark.test_stream)):
                model_dir = os.path.join(save_dir, f'{idx+1}.pth')
                model = avl.models.SimpleMLP(
                    num_classes=benchmark.n_classes,
                    hidden_size=40,
                    hidden_layers=2,
                    drop_rate=0.0
                )
                load_model(model_dir, model, device, idx)
                models.append(model)
        elif args.method == 'eraml':
            for idx in range(len(benchmark.test_stream)):
                model_dir = os.path.join(save_dir, f'{idx+1}.pth')
                model = FeatureExtractorMLP(
                    input_size=28*28,
                    hidden_size=40,
                    hidden_layers=2,
                    drop_rate=0.0,
                    relu_act=True
                )
                load_model(model_dir, model, device, idx)
                models.append(model)
                
    return models

# Calculate average L2 norm of Jacobian
def jacobian_norm(model, loader, device, n_batches=32):
    model.eval()
    total, seen = 0, 0
    crit = nn.CrossEntropyLoss()
    
    for b, (x, y) in enumerate(loader):
        if b >= n_batches: 
            break
        x, y = x.to(device), y.to(device)
        x.requires_grad_(True)

        out = model(x)
        loss = crit(out, y)
        loss.backward()
        
        g = x.grad.detach().flatten(1)
        total += g.norm(p=2, dim=1).sum().item()
        seen  += g.size(0)

        model.zero_grad(set_to_none=True)
    
    return total / seen

# Approximate largest eigenvalue of Hessian using power iteration
def hessian_spectral(model, loader, device,
                     n_batches=8, max_iter=20, tol=1e-4):
    last_fc = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    W = last_fc.weight

    v = torch.randn_like(W, device=device)
    v = v / v.norm()

    crit = nn.CrossEntropyLoss()

    for _ in range(max_iter):
        hv = torch.zeros_like(W, device=device)

        for b, (x, y) in enumerate(loader):
            if b >= n_batches: break
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = crit(out, y)
            g  = torch.autograd.grad(loss, W, create_graph=True)[0]

            dot = (g * v).sum()
            hv_batch = torch.autograd.grad(dot, W, retain_graph=False)[0]
            hv += hv_batch.detach() / n_batches

            model.zero_grad(set_to_none=True)

        hv_norm = hv.norm()
        if hv_norm < tol:
            return 0.0
        v_next = hv / hv_norm
        if (v_next - v).norm() < tol:
            v = v_next
            break
        v = v_next

    lambda_max = (v * hv).sum().item()
    return lambda_max

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(device)
    device_capability = torch.cuda.get_device_capability(device)
    device_mem = torch.cuda.get_device_properties(device).total_memory / 1024 **3

    print(f"Device name: {device_name}")
    print(f"Device capability: {device_capability}")
    print(f"Device memory: {device_mem:.2f}GB")

    transform_prototypes = transform.Compose([icarl_mnist_augment_data])
    train_transform = transform.Compose([
        icarl_mnist_augment_data,
        transform.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transform.Compose([
        transform.Normalize((0.1307,), (0.3081,))
    ])

    benchmark = avl.benchmarks.SplitMNIST(
        n_experiences=5,
        return_task_id=False,
        seed=args.seed,
        fixed_class_order=list(range(10)),
        train_transform=train_transform,
        eval_transform=test_transform,
    )
    
    configs = yaml.load(open(f"configs/{args.dataset}/{args.method}.yaml", "r"), Loader=yaml.FullLoader)
    configs = create_default_args(configs, args)
    
    models = get_models(args, configs, benchmark, device)
            
    print(f"Loaded {len(models)} models")

    ds = MNIST("./data", False, download=True, transform=test_tf)
    idx = [i for i,(_,y) in enumerate(ds) if y in [0,1]]
    loader = DataLoader(Subset(ds, idx), 64, shuffle=False)

    res = {
        "Stage":[], 
        "Jacobian":[], 
        "MinDelta":[]
    }
    if args.clever: 
        res["CLEVER"] = []

    for stage, mdl in enumerate(models,1):
        print(f"\n=== Stage {stage} ===")
        jac = jacobian_norm(mdl, loader, device)
        hes = hessian_spectral(mdl, loader, device)

        print(f"  Avg. Jacobian ‖·‖₂ : {jac:.4f}")
        print(f"  Hessian λ_max      : {hes:.4f}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--method", default="icarl",
                   choices=["icarl","gdumb","replay","erace","eraml"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clever", action="store_true", help="Compute CLEVER bound as well")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
