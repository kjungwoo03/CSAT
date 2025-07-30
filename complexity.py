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

# -------------------------------------------------------------------------
# 0.  데이터 전처리 ---------------------------------------------------------
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# 1.  모델 로딩 헬퍼 --------------------------------------------------------
# -------------------------------------------------------------------------
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
    

# -------------------------------------------------------------------------
# 2-A. Jacobian 평균 L₂-노름 ----------------------------------------------
# -------------------------------------------------------------------------
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

def hessian_spectral(model, loader, device,
                     n_batches=8, max_iter=20, tol=1e-4):
    """
    Power-iteration으로 마지막 Linear 가중치의 Hessian 최대 고유값 근사.
    * 마지막 nn.Linear 계층 W ∈ ℝ^{C×d} 에 대해 H = ∇²_W ℓ
    * 매 반복마다 Hessian-vector product를 autograd 로 계산
    """
    # 마지막 Linear 모듈 찾기
    last_fc = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    W = last_fc.weight      # (C, d)

    # 무작위 초기 v
    v = torch.randn_like(W, device=device)
    v = v / v.norm()

    crit = nn.CrossEntropyLoss()

    for _ in range(max_iter):
        hv = torch.zeros_like(W, device=device)

        for b, (x, y) in enumerate(loader):
            if b >= n_batches: break
            x, y = x.to(device), y.to(device)

            # 1차 gradient g = ∂ℓ/∂W
            out = model(x)
            loss = crit(out, y)
            g  = torch.autograd.grad(loss, W, create_graph=True)[0]

            # Hv = ∂(g·v)/∂W
            dot = (g * v).sum()
            hv_batch = torch.autograd.grad(dot, W, retain_graph=False)[0]
            hv += hv_batch.detach() / n_batches   # 평균

            model.zero_grad(set_to_none=True)

        hv_norm = hv.norm()
        if hv_norm < tol:          # 0 Hessian (드물지만)
            return 0.0
        v_next = hv / hv_norm
        if (v_next - v).norm() < tol:
            v = v_next
            break
        v = v_next

    # λ_max ≈ vᵀ (Hv)
    lambda_max = (v * hv).sum().item()
    return lambda_max

# -------------------------------------------------------------------------
# 4.  main ---------------------------------------------------------------
# -------------------------------------------------------------------------
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

    # loader (class 0,1 로 제한)
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

    # visualization = False
    # if visualization:
    #     df = pd.DataFrame(res).set_index("Stage")

    #     # ----------- 시각화 ------------
    #     plt.figure(figsize=(4,3))
    #     ax1 = plt.gca()
    #     ax1.plot(df.index, df["Jacobian"], 'o-', label="Jacobian L2")
    #     ax1.set_xlabel("Stage"); ax1.set_ylabel("Avg |∇f|₂")

    #     ax2 = ax1.twinx()
    #     ax2.plot(df.index, df["MinDelta"], 's--', color='tab:red', label="DeepFool min ‖δ‖")
    #     ax2.set_ylabel("Min ‖δ‖ (L2)", color='tab:red')
    #     ax2.tick_params(axis='y', colors='tab:red')

    #     if args.clever:
    #         ax3 = ax1.twinx()
    #         ax3.spines.right.set_position(("axes", 1.15))
    #         ax3.plot(df.index, df["CLEVER"], 'd-.', color='tab:green', label="CLEVER bound")
    #         ax3.set_ylabel("CLEVER (L2)", color='tab:green')
    #         ax3.tick_params(axis='y', colors='tab:green')

    #     h1,l1 = ax1.get_legend_handles_labels()
    #     h2,l2 = ax2.get_legend_handles_labels()
    #     hs,ls = h1+h2, l1+l2
    #     if args.clever:
    #         h3,l3 = ax3.get_legend_handles_labels()
    #         hs += h3; ls += l3
    #     ax1.legend(hs, ls, loc="upper left")
    #     ax1.set_title(f"Lipschitz trend – {args.method.upper()}")

    #     plt.tight_layout()
    #     plt.savefig(f"metrics_{args.method}.png")
        # plt.close()

# -------------------------------------------------------------------------
# 5.  CLI -----------------------------------------------------------------
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--method", default="icarl",
                   choices=["icarl","gdumb","replay","erace","eraml"])
    p.add_argument("--seed", type=int, default=42)
    # # --- CLEVER 옵션 ---
    p.add_argument("--clever", action="store_true", help="Compute CLEVER bound as well")
    # p.add_argument("--clever_nsamp", type=int, default=256)
    # p.add_argument("--clever_niter", type=int, default=100)
    return p.parse_args()

# -------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    main(args)
