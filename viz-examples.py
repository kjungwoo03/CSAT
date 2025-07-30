# viz_compare.py  (Python ≥3.8)
import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
import avalanche as avl
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
# ───────── Normalization Wrapper ───────── #
class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor(std,  dtype=torch.float32).view(1,3,1,1))
    
    def forward(self, x): 
        return self.model((x - self.mean) / self.std)

# ───────── 모델 로드 ───────── #
def load_icarl_models(model_dir, n_models, dev):
    nets=[]
    for idx in range(1, n_models+1):
        net = avl.models.make_icarl_net(num_classes=100)
        net.apply(avl.models.initialize_icarl_net)
        path = os.path.join(model_dir, f"{idx}.pth")
        net.load_state_dict(torch.load(path, map_location=dev))
        net.to(dev).eval()
        print(f"Model {idx} loaded ✔")
        nets.append(net)
    return nets

# ───────── 패치 자동 탐색 ───────── #
def auto_patch_coord(orig, adv_list, ps):
    score = torch.zeros(32-ps+1, 32-ps+1)
    for adv in adv_list:
        diff = (adv - orig).abs().sum(0, keepdim=True).unsqueeze(0)   # L1
        score += F.avg_pool2d(diff, ps, 1)[0,0] * ps**2
    y, x = divmod(score.argmax().item(), 32-ps+1)
    return y,x

# ───────── 메인 ───────── #
def main(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean=np.array([0.5071,0.4867,0.4408]); std=np.array([0.2675,0.2565,0.2761])

    # 1) 원본 이미지
    tf = T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Lambda(lambda x: x[:3])])
    orig = tf(Image.open(cfg.image_path)).to(dev)  # (3,32,32)
    orig_b = orig.unsqueeze(0)
    orig_cpu = orig.cpu()

    # 2) 10개 모델 로드
    model_dir = f'./pretrained_models/{cfg.dataset}/{cfg.method}'
    nets = load_icarl_models(model_dir, 10, dev)

    # 3) adversarial 생성
    eps, steps = 8/255, 10
    adv_imgs=[]
    for net in nets:
        pgd = torchattacks.PGD(
            NormalizedModel(net, mean, std).eval().to(dev),
            eps, eps/steps, steps
        )
        adv = pgd(orig_b, torch.tensor([0], device=dev))[0].cpu()
        adv_imgs.append(adv)

    # 4) 패치 좌표 결정
    ps = cfg.patch_size
    if cfg.fixed_patch:
        y, x = cfg.patch_y, cfg.patch_x
        print(f"[Fixed] patch = ({y},{x}) size={ps}")
    else:
        y, x = auto_patch_coord(orig_cpu, adv_imgs, ps)
        print(f"[Auto]  patch = ({y},{x}) size={ps}")

    # 공통 변수
    n_imgs = 1 + len(adv_imgs)                       # 11
    n_cols = 4
    n_rows = math.ceil(n_imgs / n_cols)

    # 5) Figure 1 : 원본 + Adv 10 개
    fig1, ax1 = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    ax1 = ax1.flatten()
    full_imgs = [orig_cpu] + adv_imgs
    titles    = ["Original"] + [f"Adv {i+1}" for i in range(10)]
    for k,(im,ttl) in enumerate(zip(full_imgs, titles)):
        ax1[k].imshow(np.clip(im.permute(1,2,0).numpy(), 0, 1))
        # ax1[k].add_patch(plt.Rectangle((x,y), ps, ps, ec="red", lw=1.5, fill=False))
        ax1[k].set_title(ttl, fontsize=9); ax1[k].axis("off")
    for k in range(len(full_imgs), n_rows*n_cols):
        ax1[k].axis("off")
    plt.tight_layout(); plt.savefig("fig_full.png", dpi=250); plt.close(fig1)

    # 6) Figure 2 : (adv - orig) × scale
    sf = cfg.diff_scale
    fig2, ax2 = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    ax2 = ax2.flatten()
    diff_imgs  = [torch.zeros_like(orig_cpu)] + [(adv - orig_cpu)*sf for adv in adv_imgs]
    diff_titles = ["Zero"] + [f"(Adv-Orig)×{sf:g}" for _ in adv_imgs]
    for k,(dm,ttl) in enumerate(zip(diff_imgs, diff_titles)):
        vis = torch.clamp(dm + 0.5, 0, 1)                 # shift & clip
        ax2[k].imshow(vis.permute(1,2,0).numpy())
        # ax2[k].add_patch(plt.Rectangle((x,y), ps, ps, ec="red", lw=1.5, fill=False))
        ax2[k].set_title(ttl, fontsize=9); ax2[k].axis("off")
    for k in range(len(diff_imgs), n_rows*n_cols):
        ax2[k].axis("off")
    plt.tight_layout(); plt.savefig("fig_diff.png", dpi=250); plt.close(fig2)

    print("Saved : fig_full.png   fig_diff.png")

# ───────── CLI ───────── #
def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--method",  default="icarl")
    p.add_argument("--image_path", default="panda.png")
    p.add_argument("--patch_size", type=int, default=6)
    p.add_argument("--fixed_patch", action="store_true")
    p.add_argument("--patch_y", type=int, default=20)
    p.add_argument("--patch_x", type=int, default=11)
    p.add_argument("--diff_scale", type=float, default=10.0,
                   help="multiply (adv-orig) by this value for visualization")
    return p.parse_args()

if __name__ == "__main__":
    main(parse())
