import argparse
import numpy as np
import os
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from compute_sharp import compute_m_sharpness
from evaluation.evaluation_metrics import (
    compute_eer,
    calculate_CLLR
)
from utils import (
    set_seed,
    load_yaml,
    get_model,
    get_train_loader,
    get_valid_loader,
    get_eval_loader,
    get_criterion
)


def main(rank, config, args):
    # Initialize process group
    if config['num_gpus'] > 1:
        dist.init_process_group(backend=config['dist_config']['dist_backend'],
                                init_method=config['dist_config']['dist_url'],
                                world_size=config['num_gpus'],
                                rank=rank)
    device = torch.device(f'cuda:{rank}')
    print(f'Rank {rank}: Initialized process group.')

    bs, dataloaders = get_dataloader(args, config)

    # Initialize model
    model = get_model(config)
    model.to(device)
    resume_ckpt = os.path.join(args.exp_dir, 'ckpts', f'{args.ckpt}.pt')
    ckpt = torch.load(resume_ckpt, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
    print(f'Resume from {resume_ckpt}')
    if config['num_gpus'] > 1:
        model = DDP(model, device_ids=[rank])

    for key, dataloader in dataloaders.items():
        # Evaluation
        score_file = os.path.join(args.exp_dir, 'scores',
                                  f'{key}_{args.ckpt}.csv')
        if os.path.exists(score_file):
            print(f'{score_file} already exists!')
        else:
            evaluate(config, device, dataloader, model, score_file)
        compute_score_results(args, score_file, key)

        # Sharpness
        criterion = get_criterion(config).to(device)
        sharpness = compute_m_sharpness(model, dataloader, criterion, args.rho,
                                        device)
        print(f'm_sharpness: {sharpness:.4f}')
        with open(os.path.join(args.exp_dir, 'sharpness.txt'), 'a') as f:
            f.write(f'{key}-{args.epoch} '
                    f'm_sharpness: {sharpness:.4f} (bs={bs} rho={args.rho})\n')


def get_dataloader(args, config):
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    bs = config['batch_size']

    if args.test == 0:
        if args.eval == 'train':
            dataloader, _ = get_train_loader(config)
        elif args.eval == 'valid':
            dataloader, _ = get_valid_loader(config)
        else:
            dataloader = get_eval_loader(config, args.eval)
        return bs, {args.eval: dataloader}

    elif args.test == 1:
        # 19LAeval, increase attack numbers
        tot_attack = 13
        dataloaders = {}
        for i in range(tot_attack):
            key = f'19LA_{i + 1}attack'

            def select_add(df):
                tot_attacks = df['attack'].unique()
                tot_attacks = np.sort(tot_attacks)
                selected_attacks = set(tot_attacks[:i + 2])
                idx = df[(df['attack'].isin(selected_attacks))].index
                print(f'{key}: #attack={i}, #samples={len(idx)}')
                print(selected_attacks)
                return df[df.index.isin(idx)].reset_index(drop=True)

            dataloader = get_eval_loader(config, '19LA', select_add)
            dataloaders[key] = dataloader
        return bs, dataloaders

    elif args.test == 2:
        # SC, increase attack numbers
        tot_attack = 9
        dataloaders = {}
        for i in range(tot_attack):
            key = f'SCE_{i + 1}attack'

            def select_add(df):
                tot_attacks = df['attack'].unique()
                tot_real = len(df[df['label'] == 'bonafide'])
                tot_fake = len(df[df['label'] == 'spoof'])
                num_real = int(tot_real / len(tot_attacks))
                num_fake = int(tot_fake / len(tot_attacks))
                tot_attacks = np.sort(tot_attacks)
                # selected_attacks = set(tot_attacks[:i+2])
                # selected_attacks = set([tot_attacks[0], tot_attacks[i+1]])
                selected_attacks = set(tot_attacks[1:i + 2])
                idx0 = df[(df['label'] == 'bonafide')].index
                idx0 = np.random.choice(idx0, num_real, replace=False)
                idx = df[(df['attack'].isin(selected_attacks))].index
                idx = np.random.choice(idx, num_fake, replace=False)
                idx = np.concatenate([idx0, idx])
                print(f'{key}: #attack={i}, #samples={len(idx)}, '
                      f'#real={num_real}, #fake={num_fake}')
                return df[df.index.isin(idx)].reset_index(drop=True)

            dataloader = get_eval_loader(config, 'SCE', select_add)
            dataloaders[key] = dataloader
        return bs, dataloaders

    elif args.test == 3:
        # 21LA, channel, codec
        # ['alaw', 'ulaw', 'gsm', 'pstn', 'g722', 'opus', 'none']
        # ['ita_tx', 'sin_tx', 'loc_tx', 'mad_tx', '-']
        dataloaders = {}
        tot_codec = 7

        for i in range(tot_codec):
            key = f'21LA_{i + 1}codec'

            def select_add(df):
                tot_codec = df['codec'].unique()
                tot_codec = np.sort(tot_codec)
                selected_codec = set(tot_codec[:i + 1])
                df = df.groupby('codec').sample(frac=0.2).reset_index(drop=True)
                idx = (df['codec'].isin(selected_codec))
                print(f'{key}: #codec={i + 1}, #samples={len(idx)}')
                return df[idx].reset_index(drop=True)

            dataloader = get_eval_loader(config, '21LA', select_add)
            dataloaders[key] = dataloader

        return bs, dataloaders

    elif args.test == 4:
        # 21LA, channel, codec
        # ['alaw', 'ulaw', 'gsm', 'pstn', 'g722', 'opus', 'none']
        # ['ita_tx', 'sin_tx', 'loc_tx', 'mad_tx', '-']
        dataloaders = {}
        tot_codec = 5

        for i in range(tot_codec):
            key = f'21LA_{i + 1}trans'

            def select_add(df):
                tot_codec = df['trans'].unique()
                tot_codec = np.sort(tot_codec)
                selected_codec = set(tot_codec[:i + 1])
                df = df.groupby('trans').sample(frac=0.2).reset_index(drop=True)
                idx = (df['trans'].isin(selected_codec))
                print(f'{key}: #codec={i + 1}, #samples={len(idx)}')
                return df[idx].reset_index(drop=True)

            dataloader = get_eval_loader(config, '21LA', select_add)
            dataloaders[key] = dataloader

        return bs, dataloaders

    elif args.test == 5:
        # SCE, increase speaker numbers
        tot_speaker = 40
        dataloaders = {}
        for i in range(1, 5):
            key = f'SCE_{i * 10}speaker'

            def select_add(df):
                tot_speaker = df['speaker'].unique()
                tot_speaker = np.sort(tot_speaker)
                selected_speaker = set(tot_speaker[:i * 10])
                print(selected_speaker)
                idx = (df['speaker'].isin(selected_speaker))
                print(f'{key}: #speaker={i * 10}, #samples={len(idx)}')
                print(df[idx]['attack'].unique())
                return df[idx].reset_index(drop=True)

            dataloader = get_eval_loader(config, 'SCE', select_add)
            dataloaders[key] = dataloader
        return bs, dataloaders


def evaluate(config, device, eval_loader, model, score_file):
    # Evaluation
    model.eval()
    files = []
    scores0 = []
    scores1 = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            x, y, file = batch
            x = x.to(device)
            y = y.to(device)
            score, _ = model(x)
            files.extend(file)
            scores0.extend(score[:, 0].cpu().numpy())
            scores1.extend(score[:, 1].cpu().numpy())
            labels.extend(y.cpu().numpy())
    scores0 = np.array(scores0)
    scores1 = np.array(scores1)
    labels = np.array(labels)
    # Save scores
    with open(score_file, 'a') as f:
        for i in range(len(files)):
            id = files[i]
            label = "bonafide" if labels[i] == 1 else "spoof"
            f.write(f'{id} {label} {scores0[i]} {scores1[i]}  \n')
    print(f'Saved scores to {score_file}')


def compute_score_results(args, score_file, key):
    result_file = os.path.join(args.exp_dir, 'sharpness.txt')

    # Load scores
    scores = pd.read_csv(score_file, header=None, sep=' ')
    scores = scores.drop_duplicates(subset=0, keep='last')
    print(f'Test score file: {score_file}, #scores: {len(scores)}')
    labels = scores[1].values
    labels = [1 if label == 'bonafide' else 0 for label in labels]
    scores0 = scores[2].values
    scores1 = scores[3].values
    labels = np.array(labels)
    scores0 = np.array(scores0)
    scores1 = np.array(scores1)
    score = np.stack([scores0, scores1], axis=1)

    # Compute EER
    eer, thresh = compute_eer(scores0, labels)
    print(f'{key}-{args.ckpt} EER: {eer:.4f}')

    # Compute cllr
    target_score = np.array([score[i, labels[i]] for i in range(len(labels))])
    nontarget_score = np.array(
        [score[i, 1 - labels[i]] for i in range(len(labels))])
    cllr = calculate_CLLR(target_score, nontarget_score)
    print(f'{key}-{args.ckpt} CLLR: {cllr:.4f}')

    # Save results
    with open(result_file, 'a') as f:
        f.write(f'{key}-{args.ckpt} EER: {eer:.4f}, CLLR: {cllr:.4f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='exp/debug')
    parser.add_argument('--epoch', type=str, default='best')
    parser.add_argument('--eval', default='19LA')
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument('--url', type=int, default=1423)
    args = parser.parse_args()

    # Load config
    config = load_yaml(os.path.join(args.exp_dir, 'config.yaml'))
    config['dist_config']['dist_url'] = f'tcp://localhost:{args.url}'
    set_seed(config['seed'])
    config['split_id'] = True
    if args.eval in {'SCD', 'SCE'}:
        config['split_id'] = False

    # Set score file
    if args.epoch.isdigit():
        args.ckpt = f"epoch{args.epoch}"
    else:
        args.ckpt = args.epoch
    os.makedirs(os.path.join(args.exp_dir, 'scores'), exist_ok=True)

    # Set environment
    assert torch.cuda.is_available(), 'CUDA is not available!'
    config['num_gpus'] = torch.cuda.device_count()
    if config['num_gpus'] > 1:
        rank = int(os.environ["SLURM_PROCID"])
        main(rank, config, args)
    else:
        main(0, config, args)
