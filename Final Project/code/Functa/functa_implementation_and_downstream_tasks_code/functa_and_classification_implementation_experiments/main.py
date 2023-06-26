import numpy as np
import paddle
# !pip install paddlepaddle-gpu
# !conda install cudatoolkit=10.2
paddle.get_device()
paddle.set_device("gpu:0")
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import sys

from model import LatentModulatedSiren # 这个就是我们的模型F_theta
from utils import get_coordinate_grid

from train import inner_step # 内圈循环、外圈循环，都写在这里
import dataloaders.mnist as mnist # paddlevision的mnist数据集，进行/255操作，并pad 至32x32

def main(model_cfg, dataset, state_dict, bias):
    bs = 9 # batch size
    recon_shape = 64 # 重建的图像尺寸，也就是二维网格I的尺寸
    inner_steps = 3
    inner_lr = 1e-2 # Follow原作超参设置
    
    # 取几个样本
    image_list = []
    coords_list = []
    # for i in range(model_cfg['batch_size']):
    for k in range(2500): # 10000 / model_cfg['batch_size'] = 2500
        images = []
        coords = []
        for i in range(model_cfg['batch_size']): # 10000 test samples
            # i, c, _, _ = dataset.__getitem__(4650 + i)
            i, c, _, _ = dataset.__getitem__(k * 4 + i)
            images.append(i)
            coords.append(c)
        images = paddle.to_tensor(np.stack(images))
        coords = paddle.to_tensor(np.stack(coords))
        image_list.append(images)
        coords_list.append(coords)
    
    # save training history
    inner_loss_list = []
    psnr_list = []
    modulate_batch_list = []
    modulate_list = []
    pp_out_list = []
    
    # 迭代3iters就可以取得不错的效果
    for k in range(20):
        model = LatentModulatedSiren(**model_cfg)
        model.set_state_dict(state_dict)

        criterion = paddle.nn.MSELoss('none') # 自定义逻辑，重建结果样本内mean，样本间sum
        inner_optim = paddle.optimizer.SGD(inner_lr, parameters=[model.latent.latent_vector]) # 内圈优化只调整phi
        
        # 初始化phi_j = [0] * 512
        paddle.assign(
            np.zeros(model.latent.latent_vector.shape).astype("float32"),
            model.latent.latent_vector,
        )
        model.train()

        for j in range(inner_steps):
            inner_loss = inner_step(image_list[20 * bias + k], coords_list[20 * bias + k], model, inner_optim, criterion)
            psnr = -10 * np.log10(inner_loss / model_cfg['batch_size'])
            # print("Inner loss {:.6f}, psnr {:.6f}".format(inner_loss[0], psnr[0]))
        inner_loss_list.append(inner_loss[0])
        psnr_list.append(psnr[0])
        modulate = model.latent.latent_vector.detach() # detach
        modulate_batch_list.append(modulate)
        for bs_i in range(model_cfg['batch_size']):
            modulate_list.append(modulate[bs_i])

        model.eval()
        print(len(modulate_batch_list))
        # print(len(psnr_list))
        pp_out = model(coords_list[20 * bias + k], modulate_batch_list[k]) # 推理
        pp_out_list.append(pp_out)
    
    # write training history to file
    with open("inner_loss.txt", "a") as file:
        for item in inner_loss_list:
            file.write(str(item) + "\n")
    with open("psnr.txt", "a") as file:
        for item in psnr_list:
            file.write(str(item) + "\n")
    with open("modulate.txt", "a") as file:
        for item in modulate_list:
            file.write(str(item) + "\n")
    with open("pp_out.txt", "a") as file:
        for item in pp_out_list:
            file.write(str(item) + "\n")
    with open("modulate_batch.txt", "a") as file:
        for item in modulate_batch_list:
            file.write(str(item) + "\n")
    

if __name__ == "__main__":
    model_cfg = {
        'batch_size': 4,
        'out_channels': 1, # Gray
        'depth': 15,
        'latent_dim': 512,
        'latent_init_scale': 0.01,
        'layer_sizes': [],
        'meta_sgd_clip_range': [0, 1],
        'meta_sgd_init_range': [0.005, 0.1],
        'modulate_scale': False,
        'modulate_shift': True,
        'use_meta_sgd': True,
        'w0': 30,
        'width': 512
    }
    
    dataset = mnist.Mnist(split="test", transforms=None) # 训练集用于优化theta
    state_dict = paddle.load("/home/zhsyy/fyx/paddle_functa/work/assets/mnist_params_512_latents_100000.pdparams")
    
    bias = int(sys.argv[2])
    # bias = 0
    main(model_cfg, dataset, state_dict, bias)