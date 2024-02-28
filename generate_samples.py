import argparse
import os.path
from copy import deepcopy
import tqdm
from model import DeformAwareGenerator, TPSSpatialTransformer, RTSpatialTransformer
import torch
import torchvision
import torchvision.transforms as transforms
from util import str2list, str2bool, load_pretrained_style_generator
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--style', type=str, default='style1')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--n_sample', type=int, default=1)
    parser.add_argument('--n_each', type=int, default=6)
    parser.add_argument('--nrow', type=int, default=0)
    parser.add_argument('--swap_res', type=str, default='32,64')
    parser.add_argument('--swap_gs', type=str, default='10,10')
    parser.add_argument('--use_stn', type=str2bool, default=True)
    parser.add_argument('--use_rtstn', type=str2bool, default=True)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--style_only', type=str2bool, default=False)
    args = parser.parse_args()

    device = args.device
    n_sample = args.n_sample
    seed = args.seed
    torch.manual_seed(seed)
    rt_swap_resolutions = tps_swap_resolutions = str2list(args.swap_res)
    swap_grid_sizes = str2list(args.swap_gs)
    if args.nrow == 0:
        args.nrow = args.n_each

    transform = transforms.Compose([transforms.Resize((1024, 1024)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    resize = transforms.Resize((args.size, args.size))

    # Load generators
    original_generator = DeformAwareGenerator(1024, 512, 8, 2, resolutions=tps_swap_resolutions,
                                              rt_resolutions=rt_swap_resolutions).to(device)

    ckpt = torch.load('checkpoints/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    original_generator.load_state_dict(ckpt["g_ema"], strict=True)
    mean_latent = original_generator.mean_latent(10000)

    generator = deepcopy(original_generator)
    stns = TPSSpatialTransformer(
        grid_size=swap_grid_sizes, resolutions=tps_swap_resolutions).to(device) if args.use_stn else None
    rt_stns = RTSpatialTransformer(resolutions=rt_swap_resolutions).to(device) if args.use_rtstn else None

    print(f'Loading generator from checkpoint: checkpoints/{args.style}.pt')
    load_pretrained_style_generator(f'./checkpoints/{args.style}.pt', generator, stns, rt_stns)

    if args.output is None:
        args.output = r'./outputs/generate'
    os.makedirs(args.output, exist_ok=True)

    for i in tqdm.tqdm(range(n_sample)):
        with torch.no_grad():
            generator.eval()
            z = torch.randn(args.n_each, 512, device=device)
            
            if not args.style_only:
                original_sample, _ = original_generator([z], truncation=0.7, truncation_latent=mean_latent)
            sample, _ = generator([z], truncation=0.7, truncation_latent=mean_latent,
                                  stns=stns, rt_stns=rt_stns, alpha=args.alpha)
            idx = f"{i}".zfill(4)
            results = [sample] if args.style_only else [original_sample, sample]
            torchvision.utils.save_image(resize(torch.cat(results)),
                                         f'{args.output}/{args.style}_{args.seed}_{args.n_each}_{idx}.png',
                                         normalize=True, value_range=(-1, 1), nrow=args.nrow)
