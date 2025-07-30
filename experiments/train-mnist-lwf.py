import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
import torchvision.transforms as transforms
import argparse
import avalanche as avl
import yaml
import os
import numpy as np
from utils import set_seed, create_default_args

class LwFCEPenalty(avl.training.LwF):
    """This wrapper around LwF computes the total loss
    by diminishing the cross-entropy contribution over time,
    as per the paper
    "Three scenarios for continual learning" by van de Ven et. al. (2018).
    https://arxiv.org/pdf/1904.07734.pdf
    The loss is L_tot = (1/n_exp_so_far) * L_cross_entropy +
                        alpha[current_exp] * L_distillation
    """
    def _before_backward(self, **kwargs):
        self.loss *= float(1/(self.clock.train_exp_counter+1))
        super()._before_backward(**kwargs)
        
        
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {torch.cuda.get_device_name(device)}")

    train_transform = transforms.Compose([
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
    
    model = avl.models.SimpleMLP(
        num_classes=benchmark.n_classes,
        hidden_size=configs.hidden_size,
        hidden_layers=configs.hidden_layers,
        drop_rate=0.0
    ).to(device)
    criterion = CrossEntropyLoss()
    
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
    
    cl_strategy = LwFCEPenalty(
        model, 
        SGD(model.parameters(), lr=configs.learning_rate), 
        criterion,
        alpha=configs.lwf_alpha, 
        temperature=configs.lwf_temperature,
        train_mb_size=configs.train_mb_size, 
        train_epochs=configs.epochs,
        device=device, 
        evaluator=eval_plugin
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
        torch.save(cl_strategy.model.state_dict(), model_filename)
        print(f'Model saved as {model_filename}')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(benchmark.test_stream, num_workers=workersNum))
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--method", type=str, default='lwf')
    return parser.parse_args()


if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    # set seed
    set_seed(args.seed)
    
    main(args)