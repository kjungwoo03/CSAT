import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
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

    
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(device)
    device_capability = torch.cuda.get_device_capability(device)
    device_mem = torch.cuda.get_device_properties(device).total_memory / 1024 **3

    print(f"Device name: {device_name}")
    print(f"Device capability: {device_capability}")
    print(f"Device memory: {device_mem:.2f}GB")

    transform_prototypes = transforms.Compose(
        [
            icarl_mnist_augment_data,
        ]
    )
    train_transform = transforms.Compose([
        icarl_mnist_augment_data,
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    benchmark = avl.benchmarks.SplitMNIST(
        n_experiences=5,
        return_task_id=False,
        fixed_class_order=list(range(10)),
        seed=args.seed,
        train_transform=train_transform,
        eval_transform=test_transform,
    )
    
    
    configs = yaml.load(open(f"configs/{args.dataset}/{args.method}.yaml", "r"), Loader=yaml.FullLoader)
    configs = create_default_args(configs, args)
    
    model: avl.models.IcarlNet = avl.models.make_icarl_net(
        num_classes=benchmark.n_classes,
        n=2,
        c=1
    )
    model.apply(avl.models.initialize_icarl_net)
    optim = SGD(
        model.parameters(),
        lr=configs.lr_base,
        momentum=0.9,
        weight_decay=configs.weight_decay
    )
    scheduler = avl.training.plugins.LRSchedulerPlugin(
        MultiStepLR(
            optim,
            milestones=configs.lr_milestones,
            gamma=1.0/configs.lr_factor
        )
    )

    loggers = [
        avl.logging.TensorboardLogger(),
        avl.logging.InteractiveLogger()
    ]
    eval_plugin = avl.training.plugins.EvaluationPlugin(
        avl.evaluation.metrics.accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        avl.evaluation.metrics.loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        avl.evaluation.metrics.timing_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        avl.evaluation.metrics.forgetting_metrics(experience=True, stream=True),
        avl.evaluation.metrics.cpu_usage_metrics(experience=True, stream=True),
        avl.evaluation.metrics.disk_usage_metrics(experience=True, stream=True),
        avl.evaluation.metrics.confusion_matrix_metrics(num_classes=benchmark.n_classes, save_image=False, stream=True),
        loggers=loggers
    )
    
    cl_strategy = avl.training.ICaRL(
        model.feature_extractor,
        model.classifier,
        optim,
        configs.memory_size,
        buffer_transform=transform_prototypes,
        fixed_memory=True,
        train_mb_size=configs.batch_size,
        train_epochs=configs.epochs,
        eval_mb_size=configs.batch_size,
        device=device,
        evaluator=eval_plugin,
        plugins=[scheduler]
    )
    
    model_dir = f'./pretrained_models/{args.dataset}/{args.method}'
    os.makedirs(model_dir, exist_ok=True)
    
    workersNum = 32
    results = []
    print(f"Starting Experiments...")
    for i, exp in enumerate(benchmark.train_stream):
        print("Start of experience, ", exp.current_experience)
        print("Current Classes: ", exp.classes_in_this_experience)

        try:
            cl_strategy.train(exp, num_workers=workersNum)
        except Exception as e:
            print(f"Error during training: {e}")

        print('Training Completed!')

        # Save the model after each experience
        model_filename = os.path.join(model_dir, f'{exp.current_experience+1}.pth')
        torch.save(model.state_dict(), model_filename)
        print(f'Model saved as {model_filename}')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(benchmark.test_stream, num_workers=workersNum))
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed value")
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--method", type=str, default='icarl')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    set_seed(args.seed)

    main(args)