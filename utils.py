import random
import numpy as np
import torch
from types import SimpleNamespace

def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False


def create_default_args(args_dict, args):
    configs = SimpleNamespace()
    
    if args.dataset == 'mnist':
        if args.method == 'gdumb':
            configs.hidden_size = args_dict['hidden_size']
            configs.mem_size = args_dict['mem_size']
            configs.hidden_layers = args_dict['hidden_layers']
            configs.epochs = args_dict['epochs']
            configs.dropout = args_dict['dropout']
            configs.learning_rate = args_dict['learning_rate']
            configs.train_mb_size = args_dict['train_mb_size']
        elif args.method == 'icarl':
            configs.batch_size = args_dict['batch_size']
            configs.memory_size = args_dict['memory_size']
            configs.epochs = args_dict['epochs']
            configs.lr_base = args_dict['lr_base']
            configs.lr_milestones = args_dict['lr_milestones']
            configs.lr_factor = args_dict['lr_factor']
            configs.weight_decay = args_dict['weight_decay']
        elif args.method == 'replay':
            configs.hidden_size = args_dict['hidden_size']
            configs.hidden_layers = args_dict['hidden_layers']
            configs.epochs = args_dict['epochs']
            configs.dropout = args_dict['dropout']
            configs.learning_rate = args_dict['learning_rate']
            configs.train_mb_size = args_dict['train_mb_size']
        elif args.method == 'erace':
            configs.mem_size = args_dict['mem_size']
            configs.lr = args_dict['lr']
            configs.train_mb_size = args_dict['train_mb_size']
            configs.batch_size_mem = args_dict['batch_size_mem']
        elif args.method == 'lwf':
            configs.lwf_alpha = args_dict['lwf_alpha']
            configs.lwf_temperature = args_dict['lwf_temperature']
            configs.epochs = args_dict['epochs']
            configs.hidden_layers = args_dict['layers']
            configs.hidden_size = args_dict['hidden_size']
            configs.learning_rate = args_dict['learning_rate']
            configs.train_mb_size = args_dict['train_mb_size']
        elif args.method == 'eraml':
            configs.mem_size = args_dict['mem_size']
            configs.lr = args_dict['lr']
            configs.temp = args_dict['temp']
            configs.train_mb_size = args_dict['train_mb_size']
            configs.batch_size_mem = args_dict['batch_size_mem']
    
    if args.dataset == 'tinyimagenet':
        if args.method == 'gdumb':
            configs.hidden_size = args_dict['hidden_size']
            configs.mem_size = args_dict['mem_size']
            configs.hidden_layers = args_dict['hidden_layers']
            configs.epochs = args_dict['epochs']
            configs.dropout = args_dict['dropout']
            configs.learning_rate = args_dict['learning_rate']
            configs.train_mb_size = args_dict['train_mb_size']
            
    if args.dataset == 'cifar100':
        if args.method == 'gdumb':
            configs.hidden_size = args_dict['hidden_size']
            configs.mem_size = args_dict['mem_size']
            configs.hidden_layers = args_dict['hidden_layers']
            configs.epochs = args_dict['epochs']
            configs.dropout = args_dict['dropout']
            configs.learning_rate = args_dict['learning_rate']
            configs.train_mb_size = args_dict['train_mb_size']
        elif args.method == 'icarl':
            configs.batch_size = args_dict['batch_size']
            configs.nb_exp = args_dict['nb_exp']
            configs.memory_size = args_dict['memory_size']
            configs.epochs = args_dict['epochs']
            configs.lr_base = args_dict['lr_base']
            configs.lr_milestones = args_dict['lr_milestones']
            configs.lr_factor = args_dict['lr_factor']
            configs.weight_decay = args_dict['weight_decay']
        elif args.method == 'replay':
            configs.num_epochs = args_dict['num_epochs']
            configs.mem_size = args_dict['mem_size']
            configs.momentum = args_dict['momentum']
            configs.weight_decay = args_dict['weight_decay']
            configs.lr = args_dict['lr']
            configs.train_mb_size = args_dict['train_mb_size']
            configs.batch_size_mem = args_dict['batch_size_mem']
        elif args.method == 'erace':
            configs.mem_size = args_dict['mem_size']
            configs.lr = args_dict['lr']
            configs.train_mb_size = args_dict['train_mb_size']
            configs.batch_size_mem = args_dict['batch_size_mem']
        elif args.method == 'eraml':
            configs.mem_size = args_dict['mem_size']
            configs.lr = args_dict['lr']
            configs.temp = args_dict['temp']
            configs.train_mb_size = args_dict['train_mb_size']
            configs.batch_size_mem = args_dict['batch_size_mem']
                       
    return configs