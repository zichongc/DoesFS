import argparse
import os
from copy import deepcopy
import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from model import RTSpatialTransformer, TPSSpatialTransformer, DeformAwareGenerator
from util import str2list, str2bool, load_pretrained_style_generator
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate deformed stylized faces with different alpha.")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--style', type=str, default='style1')
    parser.add_argument('--use_stn', type=str2bool, default=True)
    parser.add_argument('--use_rtstn', type=str2bool, default=True)
    parser.add_argument('--n_sample', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2334)
    parser.add_argument('--nrow', type=int, default=1)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--n_each', type=int, default=1)
    parser.add_argument('--swap_res', type=str, default='32,64')
    parser.add_argument('--swap_gs', type=str, default='10,10')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--alpha0', type=float, default=-0.5)
    parser.add_argument('--alpha1', type=float, default=1)
    parser.add_argument('--split', type=int, default=6)
    args = parser.parse_args()

    device = args.device
    args.nrow = args.split + 1
    n_sample = args.n_sample
    seed = args.seed
    torch.manual_seed(seed)
    style = args.style
    rt_swap_resolutions = tps_swap_resolutions = str2list(args.swap_res)
    swap_grid_sizes = str2list(args.swap_gs)
    output_dir = f'outputs/control' if args.output is None else args.output
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([transforms.Resize((1024, 1024)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    resize = transforms.Resize((args.size, args.size))

    # Load generator
    original_generator = DeformAwareGenerator(1024, 512, 8, 2, resolutions=tps_swap_resolutions,
                                              rt_resolutions=rt_swap_resolutions).to(device)
    ckpt = torch.load('checkpoints/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    original_generator.load_state_dict(ckpt["g_ema"], strict=True)
    mean_latent = original_generator.mean_latent(10000)

    generator = deepcopy(original_generator)
    stns = TPSSpatialTransformer(grid_size=swap_grid_sizes, resolutions=tps_swap_resolutions).to(device)
    rt_stns = RTSpatialTransformer(resolutions=rt_swap_resolutions).to(device)

    ckpt = f'./checkpoints/{args.style}.pt'
    print(f'Loading deformable generator from checkpoint: {ckpt}')
    load_pretrained_style_generator(ckpt, generator, stns, rt_stns)

    step = (args.alpha1 - args.alpha0) / args.split
    alphas = [args.alpha0 + step * i for i in range(args.split + 1)]

    for i in tqdm.tqdm(range(n_sample)):
        with torch.no_grad():
            generator.eval()
            z = torch.randn(args.n_each, 512, device=device)

            original_sample, _ = original_generator([z], truncation=0.7, truncation_latent=mean_latent)

            controls = []
            for alpha in alphas:
                sample, _ = generator([z], truncation=0.7, truncation_latent=mean_latent,
                                      stns=stns, rt_stns=rt_stns, alpha=alpha)
                controls.append(sample)

            controls = torch.cat(controls, dim=0)
            torchvision.utils.save_image(resize(torch.cat([original_sample, controls])),
                                         f'{output_dir}/{args.style}_{seed}.png',
                                         normalize=True, value_range=(-1, 1), nrow=args.nrow+1)
