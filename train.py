import argparse
import torch
from torchvision import transforms
from PIL import Image
import time
import math
import datetime
import os
import numpy as np
from copy import deepcopy
from torch import nn, optim, autograd
from torch.backends import cudnn
from torch.nn import functional as F
from model import TPSSpatialTransformer, RTSpatialTransformer, DeformAwareGenerator, DiscriminatorPatch, Extra
from splice_utils.splice import Splice
from tqdm import tqdm
from lpips_loss import LPIPS
from util import str2list, str2bool, accumulate, requires_grad
import warnings
warnings.filterwarnings('ignore')
cudnn.benchmark = True
torch.manual_seed(3202)


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    _loss = F.softplus(-fake_pred).mean()
    return _loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * \
                (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def warp_reg_loss(warp_flow):
    dx_reg = 1 - F.cosine_similarity(warp_flow[:, :, :-1, :-1], warp_flow[:, :, :-1, 1:])
    dy_reg = 1 - F.cosine_similarity(warp_flow[:, :, :-1, :-1], warp_flow[:, :, 1:, :-1])
    return (dx_reg + dy_reg).sum()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--style', type=str, default='style1')
    parser.add_argument('--source', type=str, default='source1.png')
    parser.add_argument('--target', type=str, default='target1.png')
    parser.add_argument('--style_ref', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default='checkpoints/stylegan2-ffhq-config-f.pt')
    parser.add_argument('--lpips_dir', type=str, default='checkpoints', help='location of lpips_loss models. Used alex')
    parser.add_argument('--img_res', type=int, default=1024)
    parser.add_argument('--num_iter', type=int, default=500)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--warp_res', type=str, default='32,64')
    parser.add_argument('--warp_gs', type=str, default='10,10')
    parser.add_argument('--cross_mode', type=str, default='f')
    parser.add_argument('--within_mode', type=str, default='f')
    parser.add_argument('--cross_layers', type=str, default='5,11')
    parser.add_argument('--within_layers', type=str, default='5')

    parser.add_argument('--g_lr', type=float, default=2e-3)
    parser.add_argument('--d_lr', type=float, default=2e-3)
    parser.add_argument('--stn_lr', type=float, default=5e-6)
    parser.add_argument('--rtstn_lr', type=float, default=1e-4)
    parser.add_argument('--adv_wt', type=float, default=1)
    parser.add_argument('--a2agg_wt', type=float, default=50000.)
    parser.add_argument('--a2agr_wt', type=float, default=50000.)
    parser.add_argument('--a2b_wt', type=float, default=6)
    parser.add_argument('--warp_wt', type=float, default=1e-6)
    parser.add_argument('--use_stn', type=str2bool, default=True)
    parser.add_argument('--use_rtstn', type=str2bool, default=True)
    parser.add_argument('--tune_g', type=str2bool, default=True)
    parser.add_argument('--stn_accum', type=float, default=0.995)
    parser.add_argument('--g_accum', type=float, default=0.5 ** (32 / (10 * 1000)))
    parser.add_argument('--hp', type=int, default=1)
    parser.add_argument('--swap_layer', type=int, default=8)

    args = parser.parse_args()
    assert args.use_stn or args.tune_g, 'at least one of these two args to be `True`'

    device = args.device
    hp = args.hp
    num_iter = args.num_iter
    g_accum = args.g_accum
    stn_accum = args.stn_accum
    rt_warp_resolutions = tps_warp_resolutions = str2list(args.warp_res)
    warp_grid_sizes = str2list(args.warp_gs)
    latent_dim = 512
    mean_path_length = 0

    transform = transforms.Compose([transforms.Resize((args.img_res, args.img_res)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # initialize generator and discriminator
    original_generator = DeformAwareGenerator(args.img_res, latent_dim, 8, 2, resolutions=tps_warp_resolutions,
                                              rt_resolutions=rt_warp_resolutions).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

    original_generator.load_state_dict(ckpt["g_ema"], strict=True)
    generator = deepcopy(original_generator).eval()

    g_ema = deepcopy(original_generator).eval()
    g_module = generator
    accumulate(g_ema, generator, 0)

    discriminator = DiscriminatorPatch(args.img_res).to(device).eval()
    discriminator.load_state_dict(ckpt['d'], strict=True)

    # for patch-level adversarial loss
    extra = Extra().to(device)

    # deformation modules
    stns = TPSSpatialTransformer(resolutions=tps_warp_resolutions, grid_size=warp_grid_sizes).to(device)
    rt_stns = RTSpatialTransformer(resolutions=rt_warp_resolutions).to(device)
    stns_ema = TPSSpatialTransformer(resolutions=tps_warp_resolutions, grid_size=warp_grid_sizes).to(device)
    rt_stns_ema = RTSpatialTransformer(resolutions=rt_warp_resolutions).to(device)

    # DINO feature extractor
    splice = Splice(device=device)

    softmax = nn.Softmax(dim=0)
    mean_latent = original_generator.mean_latent(1).unsqueeze(0).repeat(1, original_generator.n_latent, 1)
    swap = [i for i in range(args.swap_layer, original_generator.n_latent)]

    # load images (aligned)
    style_path = os.path.join('data/style_images_aligned', args.target)
    style_aligned = Image.open(style_path).convert('RGB')
    style_image = transform(style_aligned).to(device).unsqueeze(0)

    real_path = os.path.join('data/style_images_aligned', args.source)
    real_aligned = Image.open(real_path).convert('RGB')
    real_image = transform(real_aligned).to(device).unsqueeze(0)

    if args.style_ref is None:
        args.style_ref = args.target
    ref_path = os.path.join('data/style_images_aligned', args.style_ref)
    ref_aligned = Image.open(ref_path).convert('RGB')
    ref_image = transform(ref_aligned).to(device).unsqueeze(0)

    # initialize optimizers
    params_d = []
    if args.tune_g:
        params_d.append({'params': generator.parameters(), 'lr': args.g_lr})
    if args.use_stn:
        params_d.append({'params': stns.parameters(), 'lr': args.stn_lr})
    if args.use_rtstn:
        params_d.append({'params': rt_stns.parameters(), 'lr': args.rtstn_lr})

    g_optim = optim.Adam(params_d, betas=(.1, 0.99))
    d_optim = optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0, 0.99))
    e_optim = optim.Adam(extra.parameters(), lr=args.d_lr, betas=(0, 0.99))

    mode_cross = args.cross_mode
    mode_within = args.within_mode
    vit_layer_id_cross = str2list(args.cross_layers)
    vit_layer_id_within = str2list(args.within_layers)

    # inverse source reference for color alignment
    w_plus_src = mean_latent.clone()
    w_plus_src.requires_grad_(True)
    params = [{'params': w_plus_src, 'lr': 2e-3}]
    optimizer = optim.Adam(params)
    loss_lpips = LPIPS(model_dir=args.lpips_dir).to(device)

    pbar = tqdm(range(300))
    for idx in pbar:
        Gw, _ = original_generator(w_plus_src, input_is_latent=True)
        l1_loss = F.l1_loss(Gw, real_image)
        lpips_loss = loss_lpips(Gw, real_image).mean()
        loss = l1_loss + lpips_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    exp_latent_src = w_plus_src.clone()

    # inverse target reference for color alignment
    w_plus_tgt = mean_latent.clone()
    w_plus_tgt.requires_grad_(True)
    params = [{'params': w_plus_tgt, 'lr': 2e-3}]
    optimizer = optim.Adam(params)
    loss_lpips = LPIPS(model_dir=args.lpips_dir).to(device)

    pbar = tqdm(range(300))
    for idx in pbar:
        Gw, _ = original_generator(w_plus_tgt, input_is_latent=True)
        l1_loss = F.l1_loss(Gw, style_image)
        lpips_loss = loss_lpips(Gw, style_image).mean()
        loss = l1_loss + lpips_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    exp_latent_tgt = w_plus_tgt.clone()

    del loss_lpips, optimizer

    # training
    with torch.no_grad():
        # for cross-domain loss (cross)
        ref_src_feat = splice.calculate_features(real_image, mode=mode_cross, layers=vit_layer_id_cross)
        ref_tgt_feat = splice.calculate_features(style_image, mode=mode_cross, layers=vit_layer_id_cross)
        ref_src_feat /= ref_src_feat.norm(dim=[1, 2], keepdim=True)
        ref_tgt_feat /= ref_tgt_feat.norm(dim=[1, 2], keepdim=True)
        d_ref = ref_tgt_feat - ref_src_feat
        d_ref_norm = d_ref / d_ref.norm(dim=[1, 2], keepdim=True)
        d_ref_norm = d_ref_norm.repeat(args.batch, 1, 1)

        # for in-domain loss (within)
        ref_src_ssim = splice.calculate_self_sim(real_image, mode=mode_within, layers=vit_layer_id_within)
        ref_tgt_ssim = splice.calculate_self_sim(style_image, mode=mode_within, layers=vit_layer_id_within)

    loss_dict = {}
    start_time = time.time()

    for idx in range(num_iter):

        # fine-tune discriminator
        requires_grad(generator, False)
        requires_grad(stns, False)
        requires_grad(rt_stns, False)
        requires_grad(discriminator, True)
        requires_grad(extra, True)

        with torch.no_grad():
            sample_w = generator.get_latent(
                torch.randn([args.batch, latent_dim]).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
            fake_img, _ = generator(sample_w, input_is_latent=True, stns=stns, rt_stns=rt_stns)

        fake_pred = discriminator(fake_img, extra=extra, flag=1, p_ind=np.random.randint(0, hp))
        real_pred = discriminator(ref_image, extra=extra, flag=1, p_ind=np.random.randint(0, hp))  # one-shot

        d_loss = d_logistic_loss(real_pred, fake_pred)
        loss_dict['d_loss'] = d_loss

        d_optim.zero_grad()
        e_optim.zero_grad()
        d_loss.backward()
        d_optim.step()
        e_optim.step()

        del d_loss

        d_regularize = idx % 10 == 0
        if d_regularize:
            real_img = ref_image.clone()
            real_img.requires_grad = True
            real_pred = discriminator(real_img, extra=extra, flag=1, p_ind=np.random.randint(0, 3))
            real_pred = real_pred.view(real_img.size(0), -1)
            real_pred = real_pred.mean(dim=1).unsqueeze(1)

            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            extra.zero_grad()

            (10 / 2 * r1_loss * 10 +
             0 * real_pred[0]).backward()

            d_optim.step()
            e_optim.step()
            loss_dict["r1"] = r1_loss
            del r1_loss

        # fine-tune generator
        requires_grad(generator, True)
        requires_grad(stns, True)
        requires_grad(rt_stns, True)
        requires_grad(discriminator, False)
        requires_grad(extra, False)

        with torch.no_grad():
            in_latent = generator.get_latent(
                torch.randn([args.batch, latent_dim]).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)

        with torch.no_grad():
            in_latent_src = in_latent.clone()
            in_latent_src[:, swap] = exp_latent_src[:, swap]
            sam_src_img, _ = original_generator(in_latent_src, input_is_latent=True)

        in_latent_tgt = in_latent.clone()
        in_latent_tgt[:, swap] = exp_latent_tgt[:, swap]
        img, warp_flows1 = generator(in_latent_tgt, input_is_latent=True, stns=stns, rt_stns=rt_stns)

        # adv loss
        img_g, warp_flows = generator(in_latent, input_is_latent=True, stns=stns, rt_stns=rt_stns)
        fake_pred = discriminator(img_g, extra=extra, flag=1, p_ind=np.random.randint(0, hp))
        g_loss = g_nonsaturating_loss(fake_pred) * args.adv_wt
        loss_dict['g_loss'] = g_loss

        # cross-domain loss
        with torch.no_grad():
            sam_src_feat_ = splice.calculate_features(sam_src_img, mode=mode_cross, layers=vit_layer_id_cross)
            sam_src_feat = sam_src_feat_ / sam_src_feat_.clone().norm(dim=[1, 2], keepdim=True)
        sam_tgt_feat_ = splice.calculate_features(img, mode=mode_cross, layers=vit_layer_id_cross)
        sam_tgt_feat = sam_tgt_feat_ / sam_tgt_feat_.clone().norm(dim=[1, 2], keepdim=True)

        d_sam = sam_tgt_feat - sam_src_feat
        d_sam_norm = d_sam / d_sam.norm(dim=[1, 2], keepdim=True)

        cross_loss = (1 - F.cosine_similarity(
            d_sam_norm.view(args.batch, -1), d_ref_norm.view(args.batch, -1)).mean()) * args.a2b_wt
        loss_dict['cross'] = cross_loss

        # in-domain consistency loss by self-similarity
        with torch.no_grad():
            sam_src_ssim = splice.calculate_self_sim(sam_src_img, mode=mode_within, layers=vit_layer_id_within)
        sam_tgt_ssim = splice.calculate_self_sim(img, mode=mode_within, layers=vit_layer_id_within)

        # generated-generated pair
        src_C1, tgt_C1 = [], []
        for sam1 in range(args.batch):
            for sam2 in range(sam1 + 1, args.batch):
                with torch.no_grad():
                    sc = F.cosine_similarity(sam_src_ssim[sam1].view(-1), sam_src_ssim[sam2].view(-1), dim=0)
                src_C1.append(sc)
                tc = F.cosine_similarity(sam_tgt_ssim[sam1].view(-1), sam_tgt_ssim[sam2].view(-1), dim=0)
                tgt_C1.append(tc)

        src_C1s = softmax(torch.stack(src_C1, dim=0))
        tgt_C1s = softmax(torch.stack(tgt_C1, dim=0))

        mse1 = ((tgt_C1s - src_C1s) ** 2)
        wt = torch.sqrt(mse1.detach()) / torch.max(torch.sqrt(mse1.detach()))
        within_loss1 = (mse1 * wt).mean() * args.a2agg_wt

        # generated-reference pair
        src_C2, tgt_C2 = [], []
        for sam in range(args.batch):
            with torch.no_grad():
                sc = F.cosine_similarity(ref_src_ssim.view(-1), sam_src_ssim[sam].view(-1), dim=0)
                src_C2.append(sc)
            tc = F.cosine_similarity(ref_tgt_ssim.view(-1), sam_tgt_ssim[sam].view(-1), dim=0)
            tgt_C2.append(tc)
        src_C2s = softmax(torch.stack(src_C2, dim=0))
        tgt_C2s = softmax(torch.stack(tgt_C2, dim=0))

        mse2 = ((tgt_C2s - src_C2s) ** 2)
        wt = torch.sqrt(mse2.detach()) / torch.max(torch.sqrt(mse2.detach()))
        within_loss2 = (mse2 * wt).mean() * args.a2agr_wt

        within_loss = within_loss1 + within_loss2
        loss_dict['within'] = within_loss

        # warp reg
        warp_loss = 0.
        for flow in warp_flows:
            warp_loss += warp_reg_loss(flow)
        for flow in warp_flows1:
            warp_loss += warp_reg_loss(flow)
        warp_loss *= args.warp_wt
        loss_dict['warp_loss'] = warp_loss

        loss = cross_loss + within_loss + warp_loss + g_loss
        loss_dict['loss'] = loss
        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

        del cross_loss, within_loss, warp_loss, g_loss

        g_regularize = idx % 10 == 0
        if g_regularize:
            path_batch_size = 2

            latents = generator.get_latent(
                torch.randn([args.batch, latent_dim]).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
            fake_img, _ = generator(latents, input_is_latent=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(fake_img, latents, mean_path_length)

            generator.zero_grad()
            weighted_path_loss = 2 * 10 * path_loss
            weighted_path_loss.backward()
            g_optim.step()

        accumulate(g_ema, g_module, g_accum)
        accumulate(stns_ema, stns, stn_accum)
        accumulate(rt_stns_ema, rt_stns, stn_accum)

        if (idx + 1) % 50 == 0:
            print(f'[{idx + 1}/{num_iter}]', end=' ')
            for k in loss_dict.keys():
                print(f'{k}={loss_dict[k]:.8f},', end=' ')

            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print(f'Elapsed [{elapsed}]')

    os.makedirs('./outputs/models', exist_ok=True)
    torch.save({'g': g_ema.state_dict(),
                'stns': stns_ema.state_dict(),
                'rtstn': rt_stns_ema.state_dict()}, f'./outputs/models/{args.style}.pt')
