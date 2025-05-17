import time
import argparse
import torch
import numpy as np
import os
import torchvision
from PIL import Image
from torch import nn
from tqdm import tqdm
import config_zero as config
from torch.multiprocessing import Process
import torch.distributed as dist
from backbones.NSCNpp.ncsnpp_generator_adagn import NCSNpp
from backbones.fractal_diffusion_model import FractalDiffusionModel
from dataset import GetDataset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.fft

#%%
class CFDiffusionModel(nn.Module):
    def __init__(self, args, model_list):
        super().__init__()
        self.args = args
        self.models = nn.ModuleList(model_list)
        self.num_bands = args.num_bands
        # self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        # self.idwt = DWTInverse(wave='haar', mode='zero')
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        if self.args.frequency_i:
            self.alpha = nn.Parameter(torch.tensor(self.args.test_alpha).unsqueeze(0).repeat(self.args.fractal_num, 1))
            self.beta = nn.Parameter(torch.tensor(self.args.test_beta).unsqueeze(0).repeat(self.args.fractal_num, 1))
        else:
            self.alpha = nn.Parameter(torch.tensor(self.args.test_alpha))
            self.beta = nn.Parameter(torch.tensor(self.args.test_beta))

    def get_wavelet_noise_no_res(self, x_0, x_t, t):
        x0_l, x0_h = self.dwt(x_0)
        xt_l, xt_h = self.dwt(x_t)
        x0_l = self.alpha[0][t] * x0_l + (1 - self.beta[0][t]) * xt_l
        new_x0_h = []
        for i in range(len(x0_h)):
            dir0 = self.alpha[1][t] * x0_h[i][:, :, 0, :, :] + (1 - self.beta[1][t]) * xt_h[i][:, :, 0, :, :]
            dir1 = self.alpha[2][t] * x0_h[i][:, :, 1, :, :] + (1 - self.beta[2][t]) * xt_h[i][:, :, 1, :, :]
            dir2 = self.alpha[3][t] * x0_h[i][:, :, 2, :, :] + (1 - self.beta[3][t]) * xt_h[i][:, :, 2, :, :]

            modified_h = torch.stack([dir0, dir1, dir2], dim=2)
            new_x0_h.append(modified_h)
        x_td1 = self.idwt((x0_l, new_x0_h))
        return x_td1

    def get_wavelet_noise(self, x_0, x_t, t):
        x0_l, x0_h = self.dwt(x_0)
        xt_l, xt_h = self.dwt(x_t)
        x0_l = self.alpha[0][t] * x0_l + (1 - self.beta[0][t]) * xt_l
        new_x0_h = self.alpha[1][t] * x0_h[0] + (1 - self.beta[1][t]) * xt_h[0]
        res_x0 = self.alpha[2][t] * x_0 + (1 - self.beta[2][t]) * x_t
        x_td1 = self.idwt((x0_l, [new_x0_h]))
        x_td1 = self.alpha[3][2] * x_td1 + self.alpha[3][1] * res_x0
        return x_td1

    def get_fft_noise(self, x_0, x_t, t):
        coeffs_t = torch.fft.fft2(x_t, dim=(-2, -1))
        coeffs_0 = torch.fft.fft2(x_0, dim=(-2, -1))
        x_td1 = self.alpha[t] * coeffs_0 + (1 - self.beta[t]) * coeffs_t
        x_td1 = torch.fft.ifft2(x_td1, dim=(-2, -1)).real
        return x_td1

    def get_direct_noise(self, h_t, x_t, t):
        if self.alpha.ndim == 1:
            alpha = self.alpha[t]
            beta = self.beta[t]
        else:
            alpha = self.alpha[0][t]
            beta = self.beta[0][t]
        assert x_t.shape == h_t.shape
        x_td1 = ((1 - alpha) * beta).view(-1, *([1] * (h_t.dim() - 1))) * h_t + \
                (alpha * beta).view(-1, *([1] * (h_t.dim() - 1))) * x_t
        return x_td1

    def forward(self, source):
        if args.padding:
            source_t = ts_padding(source, args.fractal_sizes[0], args.fractal_sizes[-1], args.fractal_padding[0],
                                  args.fractal_padding[-1])
        else:
            source_t = transform_size(source, args.fractal_sizes[0], args.fractal_sizes[-1])
        x_t = torch.randn_like(source_t).to(source.device)
        if self.args.sample_fixed:
            x_t = global_fixed_noise
        for t in reversed(range(self.args.fractal_num)):
            if self.args.conditional_type == "coord":
                t_fractal = generate_coordinate_tensor(source.size(0), args.fractal_sizes[t], args.fractal_sizes[0]).to(
                    x_t.device)
            else:
                t_fractal = torch.full((x_t.size(0),), t, dtype=torch.long).to(x_t.device)
            latent_z = torch.randn(x_t.size(0), self.args.z_emb_dim, device=x_t.device)

            h_t, _ = self.models[t](torch.cat((x_t, source_t), axis=1), t_fractal, latent_z)
            if self.args.frequency_i and t != 0:
                x_t = self.get_wavelet_noise(h_t, x_t, t)
            else:
                x_t = self.get_direct_noise(h_t, x_t, t)
            if t != 0:
                if args.padding:
                    source_t = ts_padding(source, args.fractal_sizes[0], args.fractal_sizes[t - 1],
                                          args.fractal_padding[0], args.fractal_padding[t - 1])
                    x_t = ts_padding(x_t, args.fractal_sizes[t], args.fractal_sizes[0], args.fractal_padding[t],
                                     args.fractal_padding[0])
                    x_t = ts_padding(x_t, args.fractal_sizes[0], args.fractal_sizes[t - 1], args.fractal_padding[0],
                                     args.fractal_padding[t - 1])
                else:
                    source_t = transform_size(source, args.fractal_sizes[0], args.fractal_sizes[t - 1])
                    x_t = transform_size(x_t, args.fractal_sizes[t], args.fractal_sizes[t - 1])
        return x_t

def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

#%% Diffusion coefficients

def extract(num, shape, device):
    num = torch.tensor([num] * shape[0]).to(device)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    num = num.reshape(*reshape)
    return num

def load_checkpoint(checkpoint_dir, mapping_network, name_of_network, epoch, device = 'cuda:0'):
    checkpoint_file = checkpoint_dir.format(name_of_network, epoch)
    ckpt = torch.load(checkpoint_file, map_location=device)
    for key in list(ckpt.keys()):
         ckpt[key[7:]] = ckpt.pop(key)
    mapping_network.load_state_dict(ckpt)
    mapping_network.eval()


global_fixed_noise = None

def get_fixed_noise(dataloader, device):
    with torch.no_grad():
        for source_data, _, _, _ in dataloader:
            source_data = source_data.to(device, non_blocking=True)
            if args.padding:
                source_data = ts_padding(source_data, args.fractal_sizes[0], args.fractal_sizes[-1], args.fractal_padding[0], args.fractal_padding[-1])
            else:
                source_data = transform_size(source_data, args.fractal_sizes[0], args.fractal_sizes[-1])
            # if args.input_channels == 3:
            #     source_data = source_data.squeeze(1)
            noises = []
            for _ in range(3):
                noises.append(torch.randn_like(source_data).to(device))
            fixed_noise = torch.stack(noises)
            fixed_noise = torch.mean(fixed_noise, dim=0).to(device)
            return fixed_noise

def get_train_noise(alpha, x_0, beta, x_T, device=torch.device("cuda:0")):
    if x_T is None:
        x_T = torch.randn_like(x_0).to(device)
    assert x_T.shape == x_0.shape
    x_t = (extract((1 - alpha) * beta, x_0.shape, device) * x_0 + extract(alpha * beta, x_0.shape, device) * x_T)
    return x_t

def get_test_noise(alpha, x_0, beta, x_t, device=torch.device("cuda:0")):
    assert x_t.shape == x_0.shape
    x_td1 = (extract((1 - alpha) * beta, x_0.shape, device) * x_0 + extract(alpha * beta, x_0.shape, device) * x_t)
    return x_td1

def generate_coordinate_tensor(initial_batch, current_size, original_size=256):

    assert original_size % current_size == 0, "Original size must be divisible by current size."
    grid = original_size // current_size
    coords_per_image = grid * grid

    indices = torch.arange(coords_per_image)
    rows = indices // grid
    cols = indices % grid
    coords = torch.stack([rows, cols], dim=1)  # [16, 2]

    return coords.repeat(initial_batch, 1)

def split_image_with_padding(x, sub_num, sub_size):
    # [n, c, h, h] -> [n*sub_num*sub_num, c, sub_size, sub_size]
    n, c, h, w = x.shape
    assert h == w, f"{h} != {w}"
    assert (h - sub_size) % (sub_num - 1) == 0 , f"{h - sub_size} % {sub_num - 1}"
    stride = (h - sub_size) // (sub_num - 1)  # stride
    # print(sub_num, sub_size, stride)
    assert (sub_num - 1) * stride + sub_size == h, f"{(sub_num - 1) * stride + sub_size} != {h}"
    x = x.unfold(2, sub_size, stride).unfold(3, sub_size, stride)  # [n, c, sub_num, sub_num, sub_size, sub_size]
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # [n, sub_num, sub_num, c, sub_size, sub_size]
    x = x.view(n * sub_num * sub_num, c, sub_size, sub_size)
    return x

def merge_images_with_padding(x, sub_num, n, output_size):
    # [n*sub_num*sub_num, c, p, p]->[n, c, output_size, output_size]
    c = x.shape[1]
    p = x.shape[-1]
    s = (output_size - p) // (sub_num - 1)
    assert (sub_num - 1) * s + p == output_size, "error merge"

    x_reshaped = x.view(n, sub_num, sub_num, c, p, p)
    output = torch.zeros(n, c, output_size, output_size, device=x.device)
    count = torch.zeros(n, c, output_size, output_size, device=x.device)

    for b in range(n):
        for i in range(sub_num):
            for j in range(sub_num):
                top = i * s
                left = j * s
                sub = x_reshaped[b, i, j]  # [c, p, p]
                output[b, :, top:top + p, left:left + p] += sub
                count[b, :, top:top + p, left:left + p] += 1
    output /= count.clamp(min=1)
    return output

def ts_padding(x, input_size, output_size, input_padding_size=0, output_padding_size=0):
    valid_sizes = {32, 64, 128, 256}
    assert input_size in valid_sizes and output_size in valid_sizes, "size == 32/64/128/256"
    if input_size == output_size:
        return x
    if output_size > input_size: # merge
        sub_num = output_size // input_size
        n = x.shape[0] // (sub_num * sub_num)
        return merge_images_with_padding(x, sub_num=sub_num, n=n, output_size=output_size)
    else: # split
        sub_num = input_size // output_size
        sub_size = output_size + output_padding_size
        return split_image_with_padding(x, sub_num=sub_num, sub_size=sub_size)

def transform_size(x, input_size, output_size):
    valid_sizes = {32, 64, 128, 256}
    assert input_size in valid_sizes and output_size in valid_sizes, "size == 32/64/128/256"
    if input_size == output_size:
        return x
    if output_size > input_size:
        grid = output_size // input_size
        num_blocks = grid ** 2
        assert x.shape[0] % num_blocks == 0, f"{x.shape[0]} % {num_blocks}"
        n = x.shape[0] // num_blocks
        c = x.shape[1]
        # [n, grid, grid, c, sz, sz] -> [n, c, grid*sz, grid*sz]
        return x.view(n, grid, grid, c, input_size, input_size) \
            .permute(0, 3, 1, 4, 2, 5) \
            .contiguous().view(n, c, output_size, output_size)
    else:
        grid = input_size // output_size
        assert x.size(2) == input_size and x.size(3) == input_size
        # [n, c, grid, grid, sz, sz] -> [n*grid^2, c, sz, sz]
        return x.unfold(2, output_size, output_size) \
            .unfold(3, output_size, output_size) \
            .permute(0, 2, 3, 1, 4, 5) \
            .contiguous().view(-1, x.size(1), output_size, output_size)


def estimate_spectral_norm(model_list, source, num_iterations=10, eps=1e-6):
    spectral_norms = []  # 保存每个阶段的谱范数
    # 初始化输入
    source_t = transform_size(source, args.fractal_sizes[0], args.fractal_sizes[-1])
    x_t = torch.randn_like(source_t).to(source.device)

    with torch.no_grad(), torch.autograd.set_grad_enabled(True):  # 启用梯度计算
        for t in reversed(range(args.fractal_num)):
            # 准备阶段输入（与原代码一致）
            if args.conditional_type == "coord":
                t_fractal = generate_coordinate_tensor(source.size(0), args.fractal_sizes[t], args.fractal_sizes[0]).to(
                    x_t.device)
            else:
                t_fractal = torch.full((x_t.size(0),), t, dtype=torch.long).to(x_t.device)
            latent_z = torch.zeros(x_t.size(0), args.z_emb_dim, device=x_t.device)

            # 前向传播计算原始输出
            x_t.requires_grad_(True)  # 需要梯度以计算雅可比矩阵
            h_t, _ = model_list[t](torch.cat((x_t, source_t), axis=1), t_fractal, latent_z)

            # 幂迭代法初始化
            v = torch.randn_like(h_t)  # 随机初始化方向向量
            v = v / torch.norm(v)  # 归一化为单位向量

            # 迭代估计谱范数
            for _ in range(num_iterations):
                # 计算 J * v (前向传播)
                u = h_t.flatten() @ v.flatten()  # 标量输出以计算梯度
                # 计算梯度 (J^T * u)
                grad = torch.autograd.grad(u, x_t, retain_graph=True)[0]
                # 更新 v 为归一化的梯度方向
                v = grad.detach()
                v = v / (torch.norm(v) + eps)

            # 计算最终谱范数
            sigma = torch.norm(grad).item()
            spectral_norms.append(sigma)

            # 清空梯度并进入下一阶段（与原代码一致）
            x_t = x_t.detach()
            x_t = get_test_noise(args.test_alpha[t], h_t, args.test_beta[t], x_t)
            if t != 0:
                source_t = transform_size(source, args.fractal_sizes[0], args.fractal_sizes[t - 1])
                x_t = transform_size(x_t, args.fractal_sizes[t], args.fractal_sizes[t - 1])
    print(spectral_norms)
    return spectral_norms


#%% MAIN FUNCTION
def test_CFDM(rank, gpu, args, sn_iteration=0):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:{}'.format(gpu))

    args.phase = "test"
    args.ddp = False
    args.sample_fixed = False
    test_dataset = GetDataset("test", args.input_path, args.source, args.target, dim=args.input_channels, normed=args.normed)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    # print('test data size:' + str(len(test_dataloader)))
    model_list = []
    if args.use_model_name == "FDM":
        for i in range(args.fractal_num):
            image_size = args.fractal_sizes[i]
            if args.padding:
                image_size += args.fractal_padding[i]
            model = FractalDiffusionModel(args, image_size=image_size,
                                          num_channels=args.fractal_channels[i],
                                          level_channels=args.fractal_levels[i],
                                          attn_levels=args.fractal_attns[i]).to(device)
            checkpoint_file = args.checkpoint_path + "/{}_{}.pth"
            load_checkpoint(checkpoint_file, model, '{}_{}[{}]'.format(args.network_type, args.use_model_name, i), epoch=str(args.which_epoch), device=device)
            model_list.append(model)
            if args.ddp:
                broadcast_params(model_list[i].parameters())

    from itertools import islice

    # 使用islice获取第sn_iteration个元素
    i = sn_iteration
    data = next(islice(test_dataloader, i, i + 1))

    source_data, target_data, _, _ = data
    source_data = source_data.to(device, non_blocking=True)
    if args.input_channels == 3:
        source_data = source_data.squeeze(1)
    estimate_spectral_norm(model_list, source_data)


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port_num
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':

    import importlib.util
    import sys

    ckpts = [
        ["./checkpoints/LLVIP_vtot_FDM_logs", "config.py"],
        ["./checkpoints/LLVIP_vtot_ets_zero_FDM_logs", "config_zero.py"],
        ["./checkpoints/LLVIP_vtot_zero_y_FDM_logs", "config_zero_y.py"],
        ["./checkpoints/LLVIP_vtot_mix_FDM_logs", "config_mix.py"],
        ["./checkpoints/LLVIP_vtot_rmix_FDM_logs", "config_rmix.py"],
    ]
    for value in ckpts:
        ckpt, cfg = value
        print(cfg)
        # 已知字符串
        module_name = cfg
        module_path = cfg  # 如果 xxx.py 在当前目录下

        # 动态加载模块
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        config = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = config
        spec.loader.exec_module(config)

        parser = argparse.ArgumentParser('Diffusion Parameters')
        parser = config.load_config(parser)

        args = parser.parse_args()
        args.checkpoint_path = ckpt
        test_CFDM(0, 0, args, sn_iteration=66)

    
