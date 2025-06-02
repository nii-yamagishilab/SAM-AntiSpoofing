import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import DatasetTrain, DatasetTest
from optimizers.scheduler import LinearScheduler, CosineScheduler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(yaml_file, args=None):
    with open(yaml_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args is not None:
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
    return config


def get_model(config):
    model_name = config['model']
    if model_name == 'aasist':
        from models.aasist import AASIST
        model = AASIST()
    elif model_name == 'ssl_aasist':
        from models.ssl_aasist import SSL_AASIST
        model = SSL_AASIST(**config['ssl_config'])
    elif model_name == 'ssl_linear':
        from models.ssl_aasist import SSL_Linear
        model = SSL_Linear(**config['ssl_config'])
    else:
        raise ValueError(f'Unknown model: {model_name}')
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print(f'Initialized model: {model_name}, #params = {nb_params}')
    return model


def get_optimizer(config, model):
    if config['optimizer']['name'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     **config['optimizer']['params'])
    elif config['optimizer']['name'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    **config['optimizer']['params'])
    else:
        raise ValueError(f'Unknown optimizer: {config["optimizer"]["name"]}')
    return optimizer


def get_scheduler(config, optimizer, train_loader):
    scheduler = None
    if 'scheduler' not in config:
        return scheduler
    if config['scheduler']['name'] == 'cosine':
        from optimizers.scheduler import CosineAnnealingScheduler
        total_steps = len(train_loader) * config['num_epochs']
        scheduler = CosineAnnealingScheduler(optimizer, total_steps,
                                             **config['scheduler']['params'])
    else:
        raise ValueError(f'Unknown scheduler: {config["scheduler"]["name"]}')
    return scheduler


def get_rho_scheduler(config, iter_num):
    if 'rho_scheduler' in config:
        T_max = config['num_epochs'] * iter_num
        if config['rho_scheduler']['name'] == 'linear':
            rho_scheduler = LinearScheduler(T_max=T_max,
                                            **config['rho_scheduler']['params'])
        elif config['rho_scheduler']['name'] == 'cosine':
            rho_scheduler = CosineScheduler(T_max=T_max,
                                            **config['rho_scheduler']['params'])
        else:
            raise ValueError(f'Unknown rho scheduler: '
                             f'{config["rho_scheduler"]["name"]}')
    else:
        rho_scheduler = None
    return rho_scheduler


def get_criterion(config):
    loss_name = config['loss']['name']
    if loss_name == 'wce':
        weight = config['loss']['params']['weight']
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weight))
    else:
        raise ValueError(f'Unknown loss: {loss_name}')
    return criterion


def get_train_loader(config):
    # load dataset
    train_dataset = DatasetTrain(**config['data']['train'])
    train_sampler = None
    if config['num_gpus'] > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=config['num_workers'],
                              pin_memory=True)
    print(f'Loaded training dataset, #data={len(train_dataset)}')
    del train_dataset
    return train_loader, train_sampler


def get_valid_loader(config):
    valid_dataset = DatasetTest(**config['data']['valid'])
    valid_sampler = None
    if config['num_gpus'] > 1:
        valid_sampler = DistributedSampler(valid_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'],
                              shuffle=False,
                              sampler=valid_sampler,
                              num_workers=config['num_workers'],
                              pin_memory=True)
    print(f'Loaded validation dataset, #data={len(valid_dataset)}')
    del valid_dataset
    # additional valid set
    if 'valid1' in config['data']:
        valid_dataset1 = DatasetTest(**config['data']['valid1'])
        valid_loader1 = DataLoader(valid_dataset1,
                                   batch_size=config['batch_size'],
                                   shuffle=False,
                                   num_workers=config['num_workers'],
                                   pin_memory=True)
        print(f'Loaded validation dataset1, #data={len(valid_dataset1)}')
        del valid_dataset1
        return valid_loader, valid_loader1
    return valid_loader, None


def get_eval_loader(config, eval_set, criteria=None):
    # load dataset
    data_file_pointer = {
        '19LA': 'data/metadata/ASVspoof2019_LA_eval.csv',
        '21LA': 'data/metadata/ASVspoof2021_LA_eval.csv',
        '21DF': 'data/metadata/ASVspoof2021_DF_eval.csv',
        'ITW': 'data/metadata/In_The_Wild.csv',
        'FOR': 'data/metadata/FakeOrReal.csv',
        'WF': 'data/metadata/WaveFake.csv',
        'WFJ': 'data/metadata/WaveFake_JSUT.csv',
        'WFL': 'data/metadata/WaveFake_LJSPEECH.csv',
        'ADD': 'data/metadata/ADD2022_eval.csv',
        'SC': 'data/metadata/SpoofCeleb_evaluation.csv',
    }
    if eval_set not in data_file_pointer:
        raise ValueError(f'Unknown evaluation dataset: {eval_set}, '
                         f'please choose from {data_file_pointer.keys()}')
    data_file = data_file_pointer[eval_set]
    eval_dataset = DatasetTest(data_file=data_file,
                               select_criteria=criteria)
    eval_loader = DataLoader(eval_dataset, batch_size=config['batch_size'],
                             shuffle=False, num_workers=config['num_workers'],
                             pin_memory=True)
    print(f'Loaded evaluation dataset {eval_set}, #data={len(eval_dataset)}')
    del eval_dataset
    return eval_loader
