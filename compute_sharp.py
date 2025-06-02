import argparse
import os

import torch
from tqdm import tqdm

from utils import (
    load_yaml,
    set_seed,
    get_model,
    get_criterion,
    get_train_loader,
    get_valid_loader,
    get_eval_loader)


def compute_m_sharpness(model, dataloader, criterion, rho, device,
                        adaptive=False):
    model = model.to(device)
    model.eval()

    # Accumulate sharpness across the dataset
    total_sharpness = 0.0
    total_samples = 0

    for batch in tqdm(dataloader):

        inputs, targets = batch[0], batch[1]
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute gradients and perturb parameters
        perturbed_params = []
        for param in model.parameters():
            if param.requires_grad:
                param.grad = None  # Reset gradient

        outputs, _ = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)
        loss.backward()  # Backward pass to compute gradients

        grad_norm = []
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                if adaptive:
                    grad = torch.abs(param) * grad
                grad_norm.append(grad.norm(p=2))
        grad_norm = torch.stack(grad_norm).norm(p=2)
        scale = rho / (grad_norm + 1e-12)

        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                perturbation = scale * param.grad  # Compute perturbation
                if adaptive:
                    perturbation = torch.pow(param, 2) * perturbation
                perturbed_params.append(
                    (param, param.data.clone()))  # Save original parameters
                param.data.add_(perturbation)  # Perturb parameters

        # Evaluate the perturbed model's performance
        with torch.no_grad():
            perturbed_outputs, _ = model(inputs)
            perturbed_loss = criterion(perturbed_outputs, targets)
            sharpness = perturbed_loss.item() - loss.item()  # Compute sharpness

        # Restore original parameters
        for param, original in perturbed_params:
            param.data = original

        total_sharpness += sharpness * inputs.size(0)
        total_samples += inputs.size(0)

    m_sharpness = total_sharpness / total_samples

    return m_sharpness


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='exp/debug')
    parser.add_argument('--epoch', type=str, default='best')
    parser.add_argument('--eval', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument('--adaptive', action='store_true')
    args = parser.parse_args()

    # Load config
    config = load_yaml(os.path.join(args.exp_dir, 'config.yaml'))
    set_seed(config['seed'])

    # set environment
    assert torch.cuda.is_available(), 'CUDA is not available!'
    config['num_gpus'] = torch.cuda.device_count()
    assert config['num_gpus'] == 1, 'Only support single GPU!'
    device = torch.device(f'cuda:0')

    # Load model
    model = get_model(config)
    if args.epoch.isdigit():
        args.ckpt = f"epoch{args.epoch}"
    else:
        args.ckpt = args.epoch
    resume_ckpt = os.path.join(args.exp_dir, 'ckpts', f'{args.ckpt}.pt')
    ckpt = torch.load(resume_ckpt, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
    criterion = get_criterion(config).to(device)

    # get dataloader
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    bs = config['batch_size']
    if args.eval == 'train':
        dataloader, _ = get_train_loader(config)
    elif args.eval == 'valid':
        dataloader, _ = get_valid_loader(config)
    else:
        dataloader = get_eval_loader(config, args.eval)

    # Compute m-sharpness
    m_sharpness = compute_m_sharpness(model, dataloader, criterion,
                                      args.rho, device,
                                      adaptive=args.adaptive)
    name = 'ma-sharpness' if args.adaptive else 'm-sharpness'
    print(f'{name}: {m_sharpness:.4f}')

    # Save m-sharpness to file
    with open(os.path.join(args.exp_dir, 'sharpness.txt'), 'a') as f:
        f.write(f'{args.eval}-{args.epoch} '
                f'{name}: {m_sharpness:.4f} (bs={bs} rho={args.rho})\n')
