import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch.nn

import loss_landscapes.metrics
from utils import (
    load_yaml,
    set_seed,
    get_model,
    get_criterion,
    get_train_loader,
    get_valid_loader,
    get_eval_loader
)


def plot_1d(loss_data, steps, suffix):
    plt.figure()
    plt.plot([1 / steps * i for i in range(steps)], loss_data)
    plt.title('Linear Interpolation of Loss')
    plt.xlabel('Interpolation Coefficient')
    plt.ylabel('Loss')
    axes = plt.gca()
    plt.savefig(os.path.join(fig_dir, f'loss_{suffix}.png'))


def plot_2d(loss_data_fin, steps, suffix):
    # Contour plot
    plt.figure()
    plt.contour(loss_data_fin, levels=50)
    plt.title('Loss Contours around Trained Model')
    plt.savefig(os.path.join(fig_dir, f'contour_{suffix}.png'))

    # Flat 3D plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = np.array([[j for j in range(steps)] for i in range(steps)])
    Y = np.array([[i for _ in range(steps)] for i in range(steps)])
    ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis',
                    edgecolor='none')
    ax.set_title('Surface Plot of Loss Landscape')
    plt.savefig(os.path.join(fig_dir, f'surface_{suffix}.png'))


def plot_3d(loss_data_fin, steps, suffix):
    X = np.array([[j for j in range(steps)] for i in range(steps)])
    Y = np.array([[i for _ in range(steps)] for i in range(steps)])
    Z = loss_data_fin
    fig = go.Figure()
    # add surface plot
    fig.add_trace(
        go.Surface(
            z=Z,
            x=X,
            y=Y,
            colorscale='Viridis',
            showscale=True
        )
    )
    # Add vertical red line at the center
    center = args.step // 2
    min_value = np.min(loss_data_fin)
    max_value = np.max(loss_data_fin)
    fig.add_trace(
        go.Scatter3d(
            x=[center, center],
            y=[center, center],
            z=[min_value, max_value],
            mode='lines+markers',
            line=dict(color='red', width=4),
            marker=dict(size=5, color='red'),
            name='Center Line'
        )
    )
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title="Step",
            yaxis_title="Step",
            zaxis_title="Loss",
        ),
        title="Interactive Surface Plot",
        margin=dict(l=0, r=0, b=0, t=40),
    )
    # Save as an interactive HTML file
    fig.write_html(os.path.join(fig_dir, f"3D_{suffix}.html"))


if __name__ == '__main__':
    matplotlib.rcParams['figure.figsize'] = [18, 12]

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='exp/debug')
    parser.add_argument('--epoch', type=str, default='best')
    parser.add_argument('--eval', default='19LA')
    parser.add_argument('--step', type=int, default=20)
    parser.add_argument('--ratio', type=float, default=0.01)
    parser.add_argument('--distance', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_ckpt', action='store_true')
    args = parser.parse_args()

    # Load config
    config = load_yaml(os.path.join(args.exp_dir, 'config.yaml'))
    config['batch_size'] = 32
    set_seed(args.seed)
    fig_dir = os.path.join(args.exp_dir, 'loss')
    tmp_dir = os.path.join(args.exp_dir, 'loss', 'tmp')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # Set environment
    assert torch.cuda.is_available(), 'CUDA is not available!'
    config['num_gpus'] = torch.cuda.device_count()
    assert config['num_gpus'] == 1, 'Only support single GPU!'
    device = torch.device(f'cuda:0')

    # Initialize model
    model_initial = get_model(config).to(device)
    # model_initial = copy.deepcopy(model)
    model_final = get_model(config).to(device)
    if args.epoch.isdigit():
        args.ckpt = f"epoch{args.epoch}"
    else:
        args.ckpt = args.epoch
    resume_ckpt = os.path.join(args.exp_dir, 'ckpts', f'{args.ckpt}.pt')
    ckpt = torch.load(resume_ckpt, map_location=device)
    model_final.load_state_dict(ckpt['model'])
    # model_final = copy.deepcopy(model)
    print(f'Resume from {resume_ckpt}')

    # Define loss function
    criterion = get_criterion(config).to(device)

    # Get dataloader
    if args.eval == 'train':
        dataloader, _ = get_train_loader(config)
    elif args.eval == 'valid':
        dataloader, _ = get_valid_loader(config)
    else:
        dataloader = get_eval_loader(config, args.eval)

    # Get loss metric
    if args.ratio == 0.0:  # only use one batch
        batch = iter(dataloader).__next__()
        x = batch[0].to(device)
        y = batch[1].to(device)
        print(y)
        metric = loss_landscapes.metrics.Loss(criterion, x, y)
    else:
        metric = loss_landscapes.metrics.LossBatches(criterion,
                                                     dataloader,
                                                     device,
                                                     args.ratio)

    # Get linear interpolation
    print('Computing linear interpolation of loss...')
    suffix = str.join('_', [args.eval, args.ckpt, str(args.step),
                            str(args.ratio), str(args.distance),
                            str(args.seed)])
    tmp_file = os.path.join(tmp_dir, f'loss_{suffix}.npy')
    if os.path.exists(tmp_file):
        loss_data = np.load(tmp_file)
    else:
        loss_data = loss_landscapes.linear_interpolation(model_initial,
                                                         model_final,
                                                         metric, args.step,
                                                         deepcopy_model=False)
        np.save(tmp_file, loss_data)
    plot_1d(loss_data, args.step, suffix)

    # Get planar approximation
    print('Computing random plane of loss...')
    tmp_file = os.path.join(tmp_dir, f'contour_{suffix}.npy')
    if args.save_ckpt:
        ckpt_path = os.path.join(args.exp_dir, 'ckpts', f'{args.eval}.pt')
        loss_data_fin, best_results = loss_landscapes.random_plane_with_best_params(
            model_final, metric, args.distance, args.step,
            normalization='filter')
        best_param = best_results['param']
        best_loss = best_results['loss']
        best_index = best_results['index']
        print(f'best_loss: {best_loss}, best_index: {best_index}')
        np.save(tmp_file, loss_data_fin)
        state_dict = {'model': best_param}
        torch.save(state_dict, ckpt_path)
        print(f'Save best param to {ckpt_path}')
    else:
        if os.path.exists(tmp_file):
            loss_data_fin = np.load(tmp_file)
        else:
            loss_data_fin = loss_landscapes.random_plane(model_final, metric,
                                                         args.distance,
                                                         args.step,
                                                         normalization='filter',
                                                         deepcopy_model=False)
        np.save(tmp_file, loss_data_fin)
    plot_2d(loss_data_fin, args.step, suffix)
    plot_3d(loss_data_fin, args.step, suffix)
