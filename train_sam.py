import argparse
import os
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from evaluation.evaluation_metrics import compute_eer
from optimizers.bypass_bn import enable_running_stats, disable_running_stats
from optimizers.sam import SAM
from utils import (
    set_seed,
    load_yaml,
    get_model,
    get_optimizer,
    get_scheduler,
    get_rho_scheduler,
    get_criterion,
    get_train_loader,
    get_valid_loader
)


def train(rank, config):
    # Initialize process group
    if config['num_gpus'] > 1:
        dist.init_process_group(backend=config['dist_config']['dist_backend'],
                                init_method=config['dist_config']['dist_url'],
                                world_size=config['num_gpus'],
                                rank=rank)
        print(f'Rank {rank}: Initialized process group')
    device = torch.device(f'cuda:{rank}')

    # Some directories
    exp_dir = config['exp_dir']
    ckpt_dir = os.path.join(exp_dir, 'ckpts')
    score_dir = os.path.join(exp_dir, 'scores')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(score_dir, exist_ok=True)
    print(f'Experiment dir: {exp_dir}')

    # Set up tensorboard
    if rank == 0:
        writer = SummaryWriter(os.path.join(exp_dir, 'logs'))
        with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
        print(f'Set up tensorboard and log file')

    if config['num_gpus'] > 1: dist.barrier()

    train_loader, train_sampler = get_train_loader(config)
    valid_loader, valid_loader1 = get_valid_loader(config)

    # Initialize model
    model = get_model(config)
    model.to(device)
    if config['num_gpus'] > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Initialize optimizer and scheduler
    base_optimizer = torch.optim.Adam
    optimizer = SAM(model.parameters(), base_optimizer,
                    **config['optimizer']['params'])
    scheduler = get_scheduler(config, optimizer, train_loader)
    rho_scheduler = get_rho_scheduler(config, len(train_loader))
    print(f'Initialized optimizer and scheduler')

    # Set up loss function
    criterion = get_criterion(config).to(device)
    print(f'Initialized loss function')

    # Training loop
    best_eer = 1.0
    num_epochs = config['num_epochs']
    for epoch in range(num_epochs):
        if config['num_gpus'] > 1:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        for i, (inputs, targets) in tqdm(enumerate(train_loader),
                                         total=len(train_loader),
                                         desc=f'Epoch {epoch}/{num_epochs}'):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # First forward-backward step
            enable_running_stats(model)
            preds, _ = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second forward-backward step
            disable_running_stats(model)
            preds, _ = model(inputs)
            criterion(preds, targets).backward()
            optimizer.second_step(zero_grad=True)

            train_loss += loss.item()

            if scheduler is not None:
                scheduler.step()

            if rho_scheduler is not None:
                optimizer.update_rho(rho_scheduler.step())

        train_loss /= len(train_loader)

        if config['num_gpus'] > 1: dist.barrier()

        # Validation
        model.eval()
        dev_score_file = os.path.join(score_dir, f'dev_scores.csv')
        valid_loss = validate_predict(criterion, device, model, valid_loader,
                                      dev_score_file)
        if 'valid1' in config['data']:
            dev_score_file1 = os.path.join(score_dir, f'dev_scores1.csv')
            valid_loss1 = validate_predict(criterion, device, model,
                                           valid_loader1, dev_score_file1)

        if config['num_gpus'] > 1: dist.barrier()

        if rank == 0:
            lr = optimizer.param_groups[0]['lr']
            rho = optimizer.rho
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/lr', lr, epoch)
            writer.add_scalar('train/rho', rho, epoch)
            print(f'Epoch {epoch}/{num_epochs}: '
                  f'train/loss={train_loss:.4f}, lr={lr:.6f}, rho={rho:.6f}')

            eer = compute_metrics(dev_score_file)
            print(f'val/loss={valid_loss:.4f}, val/eer={eer:.4f}')
            writer.add_scalar('val/loss', valid_loss, epoch)
            writer.add_scalar('val/eer', eer, epoch)

            if 'valid1' in config['data']:
                eer1 = compute_metrics(dev_score_file1)
                print(f'val1/loss={valid_loss1:.4f}, val1/eer={eer1:.4f}')
                writer.add_scalar('val1/loss', valid_loss1, epoch)
                writer.add_scalar('val1/eer', eer1, epoch)

                # Calculate mean eer
                eer = (eer + eer1) / 2

            # Save checkpoint
            if config['num_gpus'] > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            ckpt = {
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            if (epoch + 1) % config['save_freq'] == 0:
                torch.save(ckpt, os.path.join(ckpt_dir, f'epoch{epoch}.pt'))
                print(f'Saved checkpoint: {epoch}')

            # Save best model
            if eer < best_eer:
                best_eer = eer
                torch.save(ckpt, os.path.join(ckpt_dir, 'best.pt'))
                print(f'Saved best model: eer={eer:.4f}')

            # Save last model
            torch.save(ckpt, os.path.join(ckpt_dir, 'last.pt'))

        if config['num_gpus'] > 1: dist.barrier()


def compute_metrics(dev_score_file):
    df = pd.read_csv(dev_score_file, sep=' ', header=None)
    df = df.drop_duplicates(subset=0, keep='last')
    print(f'Validate #scores: {len(df)}')
    y_true = df[1].values
    y_pred = df[2].values
    eer, thresh = compute_eer(y_pred, y_true)
    return eer


def validate_predict(criterion, device, model, valid_loader, dev_score_file):
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for inputs, targets, utt_id in tqdm(valid_loader,
                                            desc='Validation'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds, _ = model(inputs)
            loss = criterion(preds, targets)
            valid_loss += loss.item()
            with open(dev_score_file, 'a') as f:
                for i, (u, t, p) in enumerate(zip(utt_id, targets, preds)):
                    f.write(f'{u} {t.item()} {p[1].item()}\n')
    valid_loss /= len(valid_loader)
    return valid_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/sam.yaml')
    parser.add_argument('--exp_dir', type=str, default='exp/debug')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--url', type=str, default=None)
    args = parser.parse_args()

    # Load config
    config = load_yaml(args.config, args)
    os.makedirs(args.exp_dir, exist_ok=True)
    with open(os.path.join(args.exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Set environment
    if args.seed is not None:
        config['seed'] = args.seed
    set_seed(config['seed'])
    assert torch.cuda.is_available(), 'CUDA is not available!'
    config['num_gpus'] = torch.cuda.device_count()
    print(f'Number of GPUs: {config["num_gpus"]}')
    if args.url is not None:
        config['dist_config']['dist_url'] = f'tcp://localhost:{args.url}'

    # Start training
    if config['num_gpus'] > 1:
        # mp.spawn(train, nprocs=config['num_gpus'], args=(config,))
        rank = int(os.environ['SLURM_PROCID'])
        train(rank, config)
    else:
        train(0, config)
