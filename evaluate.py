import argparse
import numpy as np
import os
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from evaluation.evaluation_metrics import (
    compute_eer,
    calculate_tDCF_EER,
    calculate_CLLR
)
from utils import (
    set_seed,
    load_yaml,
    get_model,
    get_eval_loader
)


def evaluate(rank, config):
    # Initialize process group
    if config['num_gpus'] > 1:
        dist.init_process_group(backend=config['dist_config']['dist_backend'],
                                init_method=config['dist_config']['dist_url'],
                                world_size=config['num_gpus'],
                                rank=rank)
    device = torch.device(f'cuda:{rank}')
    print(f'Rank {rank}: Initialized process group.')

    eval_loader = get_eval_loader(config, args.eval)

    # Initialize model
    model = get_model(config)
    model.to(device)
    resume_ckpt = os.path.join(args.exp_dir, 'ckpts', f'{args.ckpt}.pt')
    ckpt = torch.load(resume_ckpt, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
    print(f'Resume from {resume_ckpt}')
    if config['num_gpus'] > 1:
        model = DDP(model, device_ids=[rank])

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
            id = files[i].split('/')[-1].split('.')[0] if config[
                'split_id'] else files[i]
            label = 'bonafide' if labels[i] == 1 else 'spoof'
            f.write(f'{id} {label} {scores0[i]} {scores1[i]}  \n')
    print(f'Saved scores to {score_file}')


def compute_score_results(args, score_file):
    result_file = os.path.join(args.exp_dir, 'results.txt')
    model_path = os.path.join(args.exp_dir, 'ckpts', f'{args.ckpt}.pt')

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
    print(f'{args.eval}-{args.ckpt} EER: {eer:.4f}')

    # Compute cllr
    target_score = np.array([score[i, labels[i]] for i in range(len(labels))])
    nontarget_score = np.array(
        [score[i, 1 - labels[i]] for i in range(len(labels))])
    cllr = calculate_CLLR(target_score, nontarget_score)
    print(f'{args.eval}-{args.ckpt} CLLR: {cllr:.4f}')

    # Save results
    with open(result_file, 'a') as f:
        f.write(f'{model_path}\n')
        f.write(f'{args.eval}-{args.ckpt} EER: {eer:.4f}, CLLR: {cllr:.4f}\n')

    # Calculate min-tDCF
    if args.eval == '19LA':
        cm_score_file = score_file
        asv_score_file = 'data/asv/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt'
        eer, dcf = calculate_tDCF_EER(cm_score_file, asv_score_file,
                                      result_file)
        with open(result_file, 'a') as f:
            f.write(f'{args.eval}-{args.ckpt} '
                    f'tDCF EER: {eer:.4f}, min-tDCF: {dcf:.4f}\n')

    elif args.eval == '21LA':
        from evaluation.evaluate_2021_LA import eval_score_file
        eer, dcf = eval_score_file(score_file, 'data/asv/keys/LA', 'eval')
        with open(result_file, 'a') as f:
            f.write(f'{args.eval}-{args.ckpt} '
                    f'tDCF EER: {eer:.4f}, min-tDCF: {dcf:.4f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='exp/debug')
    parser.add_argument('--epoch', type=str, default='best')
    parser.add_argument('--eval', default='19LA')
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
        args.ckpt = f'epoch{args.epoch}'
    else:
        args.ckpt = args.epoch
    os.makedirs(os.path.join(args.exp_dir, 'scores'), exist_ok=True)
    score_file = os.path.join(args.exp_dir, 'scores',
                              f'{args.eval}_{args.ckpt}.csv')
    if os.path.exists(score_file):
        print(f'{score_file} already exists!')
    else:
        # Set environment
        assert torch.cuda.is_available(), 'CUDA is not available!'
        config['num_gpus'] = torch.cuda.device_count()
        if config['num_gpus'] > 1:
            # mp.spawn(evaluate, nprocs=config['num_gpus'], args=(config))
            rank = int(os.environ['SLURM_PROCID'])
            evaluate(rank, config)
        else:
            evaluate(0, config)

    # Evaluate, only 1 process
    compute_score_results(args, score_file)
