import argparse
import torch
import numpy as np
import os
import torchvision
from PIL import Image
from torch import nn, optim
from tqdm import tqdm
import config_zero_y as config
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.multiprocessing import Process
import torch.distributed as dist
from torchvision.models import vgg16, VGG16_Weights
from backbones.fractal_diffusion_model import FractalDiffusionModel
from dataset import GetDataset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
# import pywt
# from geomloss import SamplesLoss
# from pytorch_wavelets import DWTForward, DWTInverse
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
            source_t = ts_padding(source, args.fractal_sizes[0], args.fractal_sizes[-1], args.fractal_padding[0], args.fractal_padding[-1])
        else:
            source_t = transform_size(source, args.fractal_sizes[0], args.fractal_sizes[-1])
        x_t = torch.randn_like(source_t).to(source.device)
        if self.args.sample_fixed:
            x_t = global_fixed_noise
        for t in reversed(range(self.args.fractal_num)):
            if self.args.conditional_type == "coord":
                t_fractal = generate_coordinate_tensor(source.size(0), args.fractal_sizes[t], args.fractal_sizes[0]).to(x_t.device)
            else:
                t_fractal = torch.full((x_t.size(0),), t, dtype=torch.long).to(x_t.device)
            latent_z = torch.randn(x_t.size(0), self.args.z_emb_dim, device=x_t.device)

            h_t, _ = self.models[t](torch.cat((x_t, source_t), axis=1), t_fractal, latent_z)
            if self.args.frequency_i and t != 0:
                x_t = self.get_wavelet_noise(h_t, x_t, t)
            else :
                x_t = self.get_direct_noise(h_t, x_t, t)
            if t != 0:
                if args.padding:
                    source_t = ts_padding(source, args.fractal_sizes[0], args.fractal_sizes[t - 1], args.fractal_padding[0], args.fractal_padding[t - 1])
                    x_t = ts_padding(x_t, args.fractal_sizes[t], args.fractal_sizes[0], args.fractal_padding[t], args.fractal_padding[0])
                    x_t = ts_padding(x_t, args.fractal_sizes[0], args.fractal_sizes[t - 1], args.fractal_padding[0], args.fractal_padding[t - 1])
                else:
                    source_t = transform_size(source, args.fractal_sizes[0], args.fractal_sizes[t - 1])
                    x_t = transform_size(x_t, args.fractal_sizes[t], args.fractal_sizes[t - 1])
        return x_t


#%% Coefficients
def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

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


def sample_from_fractal_model(model_list, source):
    if args.padding:
        source_t = ts_padding(source, args.fractal_sizes[0], args.fractal_sizes[-1], args.fractal_padding[0], args.fractal_padding[-1])
    else:
        source_t = transform_size(source, args.fractal_sizes[0], args.fractal_sizes[-1])
    x_t = torch.randn_like(source_t).to(source.device)
    with torch.no_grad():
        for t in reversed(range(args.fractal_num)):
            if args.conditional_type == "coord":
                t_fractal = generate_coordinate_tensor(source.size(0), args.fractal_sizes[t], args.fractal_sizes[0]).to(x_t.device)
            else:
                t_fractal = torch.full((x_t.size(0),), t, dtype=torch.long).to(x_t.device)

            latent_z = torch.randn(x_t.size(0), args.z_emb_dim, device=x_t.device)
            h_t, _ = model_list[t](torch.cat((x_t, source_t),axis=1), t_fractal, latent_z)
            if t != 0:
                x_t = get_test_noise(args.test_alpha[t], h_t, args.test_beta[t], x_t).to(source.device)
                if args.padding:
                    source_t = ts_padding(source, args.fractal_sizes[0], args.fractal_sizes[t-1], args.fractal_padding[0], args.fractal_padding[t-1])
                    x_t = ts_padding(x_t, args.fractal_sizes[t], args.fractal_sizes[0], args.fractal_padding[t], args.fractal_padding[0])
                    x_t = ts_padding(x_t, args.fractal_sizes[0], args.fractal_sizes[t - 1], args.fractal_padding[0], args.fractal_padding[t - 1])
                else:
                    source_t = transform_size(source, args.fractal_sizes[0], args.fractal_sizes[t - 1])
                    x_t = transform_size(x_t, args.fractal_sizes[t], args.fractal_sizes[t - 1])
    return h_t

def evaluate_samples(real_data, fake_sample):
    to_range_0_1 = lambda x: (x + 1.) / 2.
    real_data = real_data.cpu().numpy()
    fake_sample = fake_sample.cpu().numpy()
    psnr_list = []
    ssim_list = []
    mae_list = []
    for i in range(real_data.shape[0]):
        real_data_i = real_data[i]
        fake_sample_i = fake_sample[i]
        real_data_i = to_range_0_1(real_data_i)
        real_data_i = real_data_i / real_data_i.max()
        fake_sample_i = to_range_0_1(fake_sample_i)
        fake_sample_i = fake_sample_i / fake_sample_i.max()
        psnr_val = psnr(real_data_i, fake_sample_i, data_range=real_data_i.max() - real_data_i.min())
        mae_val = np.mean(np.abs(real_data_i - fake_sample_i))
        if args.input_channels == 1:
            ssim_val = ssim(real_data_i[0], fake_sample_i[0], data_range=real_data_i.max() - real_data_i.min())
        elif args.input_channels == 3:
            real_data_i = np.squeeze(real_data_i).transpose(1, 2, 0)
            fake_sample_i = np.squeeze(fake_sample_i).transpose(1, 2, 0)
            ssim_val = ssim(real_data_i, fake_sample_i, channel_axis=-1, data_range=real_data_i.max() - real_data_i.min())
        else:
            raise ValueError("Unsupported number of input channels")
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val * 100)
        mae_list.append(mae_val)
    return psnr_list, ssim_list, mae_list

class FeatureModel(torch.nn.Module):
    def __init__(self, original_model):
        super(FeatureModel, self).__init__()
        self.features1 = original_model[:7]
        self.features2 = original_model[:12]

    def forward(self, x):
        return self.features1(x), self.features2(x)

def save_image(img, save_dir, phase, iteration, input_channels):
    file_path = '{}/{}({}).png'.format(save_dir, phase, str(iteration).zfill(4))
    if input_channels == 1:
        to_range_0_1 = lambda x: (x + 1.) / 2.
        img = to_range_0_1(img)
        torchvision.utils.save_image(img, file_path)
    elif input_channels == 3:
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 127.5 + 127.5).astype(np.uint8)[..., [2, 1, 0]]
        image = Image.fromarray(img)
        image.save(file_path)

def process_and_save_image(source, target, fake, input_channels, epoch, checkpoint_path):
    if input_channels == 1:
        fake_sample1 = torch.cat((source, target, fake), axis=-1)
        torchvision.utils.save_image(fake_sample1, os.path.join(checkpoint_path, f'ITER({str(epoch).zfill(4)}).png'), normalize=True)
    elif input_channels == 3:
        target = target[0].permute(1, 2, 0).cpu().numpy()
        source = source[0].permute(1, 2, 0).cpu().numpy()
        fake = fake[0].permute(1, 2, 0).cpu().numpy()
        target = (target * 127.5 + 127.5).astype(np.uint8)[..., [2, 1, 0]]
        source = (source * 127.5 + 127.5).astype(np.uint8)[..., [2, 1, 0]]
        fake = (fake * 127.5 + 127.5).astype(np.uint8)[..., [2, 1, 0]]
        fake_sample = np.concatenate((source, target, fake), axis=1)
        fake_sample = Image.fromarray(fake_sample)
        fake_sample.save(os.path.join(checkpoint_path, f'ITER({str(epoch).zfill(4)}).png'))

def process_and_save_unaligned_image(source, target, fake_source, fake_target, input_channels, epoch, checkpoint_path):
    if input_channels == 1:
        fake_sample1 = torch.cat((source, target, fake_source, fake_target), axis=-1)
        torchvision.utils.save_image(fake_sample1, os.path.join(checkpoint_path, f'ITER({str(epoch).zfill(4)}).png'), normalize=True)
    elif input_channels == 3:
        target = target[0].detach().cpu().permute(1, 2, 0).numpy()
        source = source[0].detach().cpu().permute(1, 2, 0).numpy()
        fake_source = fake_source[0].detach().cpu().permute(1, 2, 0).numpy()
        fake_target = fake_target[0].detach().cpu().permute(1, 2, 0).numpy()
        target = (target * 127.5 + 127.5).astype(np.uint8)[..., [2, 1, 0]]
        source = (source * 127.5 + 127.5).astype(np.uint8)[..., [2, 1, 0]]
        fake_source = (fake_source * 127.5 + 127.5).astype(np.uint8)[..., [2, 1, 0]]
        fake_target = (fake_target * 127.5 + 127.5).astype(np.uint8)[..., [2, 1, 0]]
        fake_sample = np.concatenate((source, target, fake_source, fake_target), axis=1)
        fake_sample = Image.fromarray(fake_sample)
        fake_sample.save(os.path.join(checkpoint_path, f'ITER({str(epoch).zfill(4)}).png'))


#%% MAIN FUNCTION
def train_CFDM(rank, gpu, args):
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    args.phase = "train"
    args.val = True
    args.sample_fixed = True
    train_dataset = GetDataset("train", args.input_path, args.source, args.target, dim=args.input_channels, normed=args.normed)
    train_dataset_indices = np.random.choice(len(train_dataset), args.cfdm_dataset_size, replace=False)
    train_dataset = Subset(train_dataset, train_dataset_indices)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=True)
    print('train data size:' + str(len(train_dataloader)))
    if args.sample_fixed:
        global global_fixed_noise
        global_fixed_noise = get_fixed_noise(train_dataloader, device)
    if args.val is True:
        val_dataset = GetDataset("val", args.input_path, args.source, args.target, dim=args.input_channels, normed=args.normed)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=val_sampler, drop_last=True)
        print('val data size:' + str(len(val_dataloader)))

    model_list = []
    feature_model = FeatureModel(vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval())
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

        cfdm = CFDiffusionModel(args=args, model_list=model_list).to(device)
        if args.ddp:
            broadcast_params(cfdm.parameters())
        params = [cfdm.alpha] + [cfdm.beta]
        optimizer_cfdm = optim.Adam(params, lr=args.cfdm_lr, betas=(args.beta1, args.beta2))
        scheduler_cfdm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cfdm, args.max_epoch, eta_min=args.cfdm_lrf)
        if args.ddp:
            cfdm = nn.parallel.DistributedDataParallel(cfdm, device_ids=[gpu])

    save_dir = args.checkpoint_path + "/generated_samples/FDM({})".format(args.which_epoch)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    global_step, epoch, init_epoch = 0, 0, 0

    for epoch in range(init_epoch, args.max_epoch + 1):

        if rank == 0:

            if args.ddp:
                tmp_alpha = cfdm.module.alpha.cpu().tolist()
                tmp_beta = cfdm.module.beta.cpu().tolist()
            else:
                tmp_alpha = cfdm.alpha.cpu().tolist()
                tmp_beta = cfdm.beta.cpu().tolist()

            print('Epoch {} Alpha : {}'.format(epoch, tmp_alpha))
            print('Epoch {} Beta : {}'.format(epoch, tmp_beta))

            if args.val is True and epoch % 1 == 0:
                val_path = os.path.join(args.checkpoint_path, 'val_({})'.format(str(epoch).zfill(4)))
                if not os.path.exists(val_path):
                    os.makedirs(val_path)

                with (torch.no_grad()):
                    PSNR = []
                    SSIM = []
                    MAE = []
                    for iteration, (source_data, target_data, _, _) in enumerate(val_dataloader):
                        target_data = target_data.to(device, non_blocking=True)
                        source_data = source_data.to(device, non_blocking=True)
                        if args.input_channels == 3:
                            target_data = target_data.squeeze(1)
                            source_data = source_data.squeeze(1)
                        fake_sample = cfdm(source_data)
                        psnr_list, ssim_list, mae_list = evaluate_samples(target_data, fake_sample)
                        PSNR.extend(psnr_list)
                        SSIM.extend(ssim_list)
                        MAE.extend(mae_list)
                    print('VAL PSNR:' + str(sum(PSNR) / len(PSNR)))
                    print('VAL SSIM:' + str(sum(SSIM) / len(SSIM)))
                    print('VAL MAE:' + str(sum(MAE) / len(MAE)))

        train_sampler.set_epoch(epoch)
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}/{args.max_epoch}", unit="iter",
                  disable=(dist.get_rank() != 0), colour='green') as pbar:
            for iteration, (source_data, target_data, _, _) in enumerate(train_dataloader):
                source_data = source_data.to(device, non_blocking=True)
                target_data = target_data.to(device, non_blocking=True)
                fake_sample = cfdm(source_data)
                compound_dist = 0
                l1 = F.l1_loss(fake_sample, target_data)
                l2 = F.mse_loss(fake_sample, target_data)
                if args.input_channels == 1:
                    target_data = target_data.repeat(1, 3 // args.input_channels, 1, 1)
                    fake_sample = fake_sample.repeat(1, 3 // args.input_channels, 1, 1)
                real_features1, real_features2 = feature_model(target_data)
                fake_features1, fake_features2 = feature_model(fake_sample)
                dist_perceptual1 = F.l1_loss(fake_features1, real_features1)
                dist_perceptual2 = F.l1_loss(fake_features2, real_features2)
                dist_perceptual = (dist_perceptual1 + dist_perceptual2) / 2.
                compound_dist += args.lambda_l1 * l1 + args.lambda_l2 * l2
                compound_dist += args.lambda_perceptual * dist_perceptual
                optimizer_cfdm.zero_grad()
                compound_dist.backward()
                optimizer_cfdm.step()
                global_step += 1
                if iteration % args.log_iteration == 0:
                    if rank == 0:
                        pbar.set_postfix(cfdm=compound_dist.item())
                pbar.update(1)

        if not args.no_lr_decay:
            scheduler_cfdm.step()


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
    parser = argparse.ArgumentParser('cell diffusion parameters')
    parser = config.load_config(parser)

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if args.network_type == 'normal':
        print("Using normal network configuration.")
    elif args.network_type == 'large':
        print("Using large network configuration.")
        args.num_channels_dae = 128
        args.num_res_blocks = 3
    elif args.network_type == 'max':
        print("Using max network configuration.")
        args.num_channels_dae = 128
        args.num_res_blocks = 4
        args.ch_mult = [1, 1, 2, 2, 4, 8]
    else:
        print(f"Unknown network type: {args.network_type}")

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train_CFDM, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        init_processes(0, size, train_CFDM, args)
    
