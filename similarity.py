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

def icarl_mnist_augment_data(img):
    img = img.numpy()
    padded = np.pad(img, ((0, 0), (2, 2), (2, 2)), mode='constant')
    random_cropped = np.zeros(img.shape, dtype=np.float32)
    crop = np.random.randint(0, high=4 + 1, size=(2,))

    # Cropping and possible flipping 
    if np.random.randint(2) > 0:
        random_cropped[:, :, :] = \
            padded[:, crop[0]:(crop[0]+28), crop[1]:(crop[1]+28)]
    else:
        random_cropped[:, :, :] = \
            padded[:, crop[0]:(crop[0]+28), crop[1]:(crop[1]+28)][:, :, ::-1]
    t = torch.tensor(random_cropped)

    return t

def load_model(model_path, model, device, model_num):
    model_filename=model_path
    state_dict = torch.load(model_filename, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    
    print(f"Model {model_num+1} loaded from {model_filename}")
    

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
                    input_size=28*28,  # MNIST 원본 크기: 784
                    hidden_size=40,
                    hidden_layers=2,
                    drop_rate=0.0,
                    relu_act=True
                )
                load_model(model_dir, model, device, idx)
                models.append(model)
                
    return models
    
    
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(device)
    device_capability = torch.cuda.get_device_capability(device)
    device_mem = torch.cuda.get_device_properties(device).total_memory / 1024 **3

    print(f"Device name: {device_name}")
    print(f"Device capability: {device_capability}")
    print(f"Device memory: {device_mem:.2f}GB")

    transform_prototypes = transform.Compose(
        [
            icarl_mnist_augment_data,
        ]
    )
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

    ######################################
    # A. Parameter-based cosine similarity
    ######################################
    print("\n--------------------------------")
    print("A. Parameter-based cosine similarity")
    def flatten_params(model):
        return parameters_to_vector([p.detach().cpu() for p in model.parameters()])

    vecs = [flatten_params(m) for m in models]
    param_cos = torch.zeros(len(models), len(models))

    for i, j in combinations(range(len(models)), 2):
        cos = torch.nn.functional.cosine_similarity(vecs[i], vecs[j], dim=0)
        param_cos[i, j] = param_cos[j, i] = cos.item()
    
    # Replace torch.fill_diagonal_ with manual diagonal fill
    for i in range(len(models)):
        param_cos[i,i] = 1.0

    print("Parameter-vector cosine matrix:\n", param_cos)
    
    ######################################
    # B. Gradient-based cosine similarity
    ######################################
    print("\n--------------------------------")
    print("B. Gradient-based cosine similarity")
    
    exp = benchmark.test_stream[0]
    dataset = exp.dataset
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
    
    def input_grad_vector(model, loader, device, num_batches=20):
        model.eval()
        grad_sum = None
        criterion = nn.CrossEntropyLoss()

        for b, (x, y, _) in enumerate(loader):
            if b >= num_batches:          # 20 미니배치 정도면 충분
                break
            x, y = x.to(device), y.to(device)
            x.requires_grad_(True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            g = x.grad.detach().view(x.size(0), -1)      # [B, 784]
            g = F.normalize(g, dim=1)                    # ℓ2 정규화
            g_batch = g.mean(0)                          # [784]

            grad_sum = g_batch if grad_sum is None else grad_sum + g_batch
            x.grad.zero_()

        return grad_sum / num_batches                    # 평균 gradient

    gvecs = [input_grad_vector(m, dataloader, device) for m in models]
    grad_cos = torch.zeros(len(models), len(models))
    for i, j in combinations(range(len(models)), 2):
        cos = F.cosine_similarity(gvecs[i], gvecs[j], dim=0)
        grad_cos[i, j] = grad_cos[j, i] = cos.item()
    grad_cos.fill_diagonal_(1.0)

    print("Gradient cosine matrix:\n", grad_cos)
    
    
    ######################################
    # C. CKA and PWCCA
    ######################################

    print("\n--------------------------------")
    print("C. CKA / PWCCA")

    def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
        """Centered Kernel Alignment for two feature matrices (linear kernel).

        Args:
            X: [n, d1]  feature matrix
            Y: [n, d2]  feature matrix (same #samples)
        Returns:
            scalar CKA value in [0, 1]
        """
        # 1) column-wise centering
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)

        # 2) Gram matrices
        dot_xy = (X.T @ Y)                     # [d1, d2]
        dot_xx = (X.T @ X)
        dot_yy = (Y.T @ Y)

        # 3) Frobenius norms
        hsic_xy = (dot_xy ** 2).sum()
        norm_xx = (dot_xx ** 2).sum().sqrt()
        norm_yy = (dot_yy ** 2).sum().sqrt()

        return (hsic_xy / (norm_xx * norm_yy)).item()
    
    def get_feats(model, loader):
        model.eval()
        feats = []
        with torch.no_grad():
            for x, _, _ in loader:
                x = x.to(device)
                if args.method == 'icarl':
                    z = model.feature_extractor(x).flatten(1)
                elif args.method == 'gdumb' or args.method == 'replay' or args.method == 'erace':
                    z = model.get_features(x)   
                elif args.method == 'eraml':
                    z = model.feature_extractor(x.view(-1, 28*28)).flatten(1)
                feats.append(z.cpu())
        return torch.cat(feats)   
    

    feat_sets = [get_feats(m, dataloader) for m in models]

    cka_mat = torch.eye(len(models))

    for i, j in combinations(range(len(models)), 2):
        cka_score = linear_cka(feat_sets[i], feat_sets[j])
        cka_mat[i, j] = cka_mat[j, i] = cka_score

    print("CKA matrix:\n", cka_mat)
    
    param_cos_np = param_cos.cpu().numpy()
    grad_cos_np = grad_cos.cpu().numpy()
    cka_mat_np = cka_mat.cpu().numpy()
    
    # Save similarity results to CSV
    results = []
    for i in range(len(models)):
        for j in range(len(models)):
            results.append({
                'Method': args.method,
                'Model_i': i+1,
                'Model_j': j+1,
                'Param_Cos': param_cos_np[i,j],
                'Grad_Cos': grad_cos_np[i,j],
                'CKA': cka_mat_np[i,j]
            })
            
    df = pd.DataFrame(results)
    df.to_csv(f'./results/similarity_{args.dataset}_{args.method}.csv', index=False)
    
    matrices = [
        (param_cos_np, "Parameter‐Cosine Similarity"),
        (grad_cos_np,  "Gradient‐Cosine Similarity"), 
        (cka_mat_np,   "CKA Similarity")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for idx, (mat, title) in enumerate(matrices):
        im = axes[idx].imshow(mat, cmap='viridis', interpolation='nearest')
        axes[idx].set_title(title)
        fig.colorbar(im, ax=axes[idx])
        axes[idx].set_xticks(np.arange(mat.shape[0]))
        axes[idx].set_yticks(np.arange(mat.shape[0]))
        axes[idx].set_xticklabels(np.arange(1, mat.shape[0] + 1))
        axes[idx].set_yticklabels(np.arange(1, mat.shape[0] + 1))
        axes[idx].set_xlabel("Stage")
        axes[idx].set_ylabel("Stage")
        
        # Add text annotations
        for i in range(mat.shape[0]):
            for j in range(mat.shape[0]):
                text = axes[idx].text(j, i, f'{mat[i, j]:.2f}',
                                    ha="center", va="center", color="w")

    plt.tight_layout()
    plt.savefig(f"./results/similarity_{args.dataset}_{args.method}.png")
    
    # stage‑to‑final similarity (stages 1‑4 vs stage 5)
    sim_param = param_cos_np[:-1, -1]
    sim_grad  = grad_cos_np[:-1, -1]
    sim_cka   = cka_mat_np[:-1,  -1]

    # -------- load ASR CSV --------
    df = pd.read_csv("./results/mnist.csv")

    # map CLI method -> csv label prefix
    csv_map = {
        "icarl": "i",    # iiCaRL
        "gdumb": "G",    # GGDumb
        "replay": "GGenReplay",
        "erace": "GER-ACE",
        "eraml": "GER-AML"
    }

    csv_key = csv_map[args.method]
    row = df[(df["Attack"] == "PGD") & (df["MMethod"].str.contains(csv_key, case=False))]
    if row.empty:
        raise ValueError("Method not found in csv")

    row = row.iloc[0]
    asr_vals = np.array([row[f"Model {i}"] for i in range(1, 6)], dtype=float)
    asr_ratio = asr_vals[:-1] / asr_vals[-1]   # stages 1‑4 divided by stage 5

    # -------- scatter plots --------
    metrics = [("Parameter Cos.", sim_param),
            ("Gradient Cos.", sim_grad),
            ("CKA",           sim_cka)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (title, x) in zip(axes, metrics):
        ax.scatter(x, asr_ratio, s=60)
        # best‑fit line
        m, b = np.polyfit(x, asr_ratio, 1)
        ax.plot(x, m*x + b, ls="--")
        # Pearson r
        r = np.corrcoef(x, asr_ratio)[0, 1]
        ax.set_title(f"{title}\n$r$={r:.2f}")
        ax.set_xlabel("Similarity")
        ax.set_ylabel("ASR Ratio (k→5)")
    plt.suptitle(f"{args.method} Similarity vs ASR Ratio")
    plt.tight_layout()
    plt.savefig(f"./results/corr_{args.dataset}_{args.method}.png")
    
    print(f"Figure saved at ./results/corr_{args.dataset}_{args.method}.png")
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--method", type=str, default='icarl')
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)