import os
import argparse

import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

import model
import utils
import dataloaders


def inner_step(images, coords, network, optim, criterion, modulate=None, meta_sgd=False):
    """
    batch_ratio: 如果推理生成modulation时bs与训练时不同，MSE loss的mean策略会导致梯度变化
        如训练时bs=4，预测时bs=1，则预测时梯度比训练时大了4倍
    """
    recon = network(coords, modulate)

    loss = criterion(recon, images) # N, H, W, C
    loss = loss.mean([1, 2, 3]).sum(0)

    loss.backward()

    if meta_sgd:
        print(network.latent.latent_vector.grad.sum().item(), network.meta_sgd_lrs().sum().item())
        with paddle.no_grad():
            meta_lr = network.meta_sgd_lrs()
            mod_grad = modulate.grad if modulate is not None else network.latent.latent_vector.grad
            paddle.assign(meta_lr * mod_grad, mod_grad)
        print(network.latent.latent_vector.grad.sum().item(), network.meta_sgd_lrs().sum().item())
    optim.step()
    optim.clear_grad()
    network.clear_gradients()

    return loss.numpy()


def outer_step(images, coords, network, optim, criterion, modulate=None):
    recon = network(coords, modulate)

    loss = criterion(recon, images)
    loss = loss.mean([1, 2, 3]).sum(0)

    loss.backward()
    optim.step()
    optim.clear_grad()
    network.clear_gradients()

    return loss.numpy()


def parse_args():
    """
    command args
    """
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        "--dset", dest="dset", help="Use dataset", default=None, type=str)

    parser.add_argument(
        "--dsetRoot", dest="dsetRoot", help="Rootpath of dataset", default=None, type=str)

    parser.add_argument(
        "--output", dest="output", help="Rootpath of output", default="./output/", type=str)

    parser.add_argument(
        "--batchsize", dest="batchsize", default=16, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    # Args
    args = parse_args()
    # Config
    batch_size = args.batchsize
    inner_steps = 3
    outer_steps = 100000
    inner_lr = 1e-2
    outer_lr = 3e-6
    latent_init_scale = 0.01
    save_interval = 100 # save ckpt interval
    ckpt_dir = args.output

    # Dataloader
    dataloader = dataloaders.create_dataloader(args.dset, args.dsetRoot, batch_size)

    # Prepare
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Model
    model_cfg = {
        'batch_size': batch_size,
        'out_channels': 3, # RGB
        'depth': 15,
        'latent_dim': 512,
        'latent_init_scale': 0.01,
        'layer_sizes': [],
        'meta_sgd_clip_range': [0, 1],
        'meta_sgd_init_range': [0.005, 0.1],
        'modulate_scale': False,
        'modulate_shift': True,
        'use_meta_sgd': False,
        'w0': 30,
        'width': 512}

    if args.dset == "mnist":
        model_cfg['out_channels'] = 1

    network = model.LatentModulatedSiren(**model_cfg)
    network.set_state_dict(paddle.load("./mnist_output/iter_400/model.pdparams"))

    # Optimizer
    ## Inner optimizer
    inner_optim = paddle.optimizer.SGD(inner_lr, 
                    parameters=[network.latent.latent_vector, network.meta_sgd_lrs.meta_sgd_lrs])
    ## Outer optimizer
    outer_optim = paddle.optimizer.Adam(outer_lr, weight_decay=1e-4,
                    parameters=[p for n, p in network.named_parameters() if n != "latent.latent_vector"])

    # Loss
    criterion = nn.MSELoss(reduction='none')

    # Multi GPU prepare
    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True, strategy=utils.get_strategy())
        inner_optim = paddle.distributed.fleet.distributed_optimizer(inner_optim)
        outer_optim = paddle.distributed.fleet.distributed_optimizer(outer_optim)
        ddp_network = paddle.DataParallel(network, find_unused_parameters=True)

    # Train loop
    iter = 0
    iterator = dataloader.__iter__()

    while iter < outer_steps:
        try:
            images, coords, labels, idxs = iterator.next()
        except StopIteration:
            iterator = dataloader.__iter__()
            images, coords, labels, idxs = iterator.next()
        iter += 1
        if iter > outer_steps: break

        paddle.assign(
            np.zeros(network.latent.latent_vector.shape).astype("float32"),
            network.latent.latent_vector,
        )

        for j in range(inner_steps):
            inner_loss = inner_step(images, coords, 
                                        ddp_network if nranks > 1 else network, inner_optim, criterion, 
                                        meta_sgd=model_cfg['use_meta_sgd'])
            if local_rank == 0:
                psnr = -10 * np.log10(inner_loss / batch_size)
                print("Outer iter {}: [{}/{}], inner loss {:.6f}, psnr {:.6f}".format(iter + 1, j + 1, 
                    inner_steps, inner_loss[0], psnr[0]))
        
        modulate = network.latent.latent_vector.detach() # detach
        outer_loss = outer_step(images, coords, ddp_network if nranks > 1 else network, outer_optim, criterion, modulate)
        if local_rank == 0:
            psnr = -10 * np.log10(outer_loss / batch_size)
            print("Outer iter {}/{}: outer loss {:.6f}, outer PSNR {:.6f}".format(iter + 1, 
                    outer_steps, outer_loss[0], psnr[0]))

        if local_rank == 0 and iter > 0 and iter % save_interval == 0:
            current_save_dir = os.path.join(ckpt_dir, "iter_{}".format(iter))
            if not os.path.exists:
                os.makedirs(current_save_dir)
            paddle.save(network.state_dict(), os.path.join(current_save_dir, 'model.pdparams'))

    iterator.__del__()