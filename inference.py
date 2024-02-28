import argparse
import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from model import DeformAwareGenerator, TPSSpatialTransformer, RTSpatialTransformer
from face_align import align_face
from e4e_projection import projection as e4e_projection
from util import str2list, str2bool, load_pretrained_style_generator
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(
    description="Generate deformed stylized faces with different alpha.")
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--style', type=str, default='style3')
parser.add_argument('--swap_res', type=str, default='32,64')
parser.add_argument('--swap_gs', type=str, default='10,10')
parser.add_argument('--input_image', type=str, default='./data/test_inputs/002.png')
parser.add_argument('--align', type=str2bool, default=True, help='set as True if input image is not aligned.')
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--alpha', type=float, default=0.8, help='deformation degree')
args = parser.parse_args()

device = args.device
rt_swap_resolutions = tps_swap_resolutions = str2list(args.swap_res)
swap_grid_sizes = str2list(args.swap_gs)
output_dir = f'outputs/inference' if args.output is None else args.output
os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([transforms.Resize((1024, 1024)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load deformable generator
generator = DeformAwareGenerator(1024, 512, 8, 2, resolutions=tps_swap_resolutions,
                                 rt_resolutions=rt_swap_resolutions).to(device)
stns = TPSSpatialTransformer(grid_size=swap_grid_sizes, resolutions=tps_swap_resolutions).to(device)
rt_stns = RTSpatialTransformer(resolutions=rt_swap_resolutions).to(device)
load_pretrained_style_generator(f'./checkpoints/{args.style}.pt', generator, stns, rt_stns)

# load input image
input_image = Image.open(args.input_image).convert('RGB')
# face align
if args.align:
    input_image = align_face(args.input_image)

# e4e inversion
latent_path = os.path.join(output_dir, 'code_' + os.path.basename(args.input_image).split('.')[0] + '.pt')
if not os.path.exists(latent_path):
    latent = e4e_projection(input_image, latent_path, device)
else:
    latent = torch.load(latent_path)['latent']

latent = latent.unsqueeze(0)

# inference
result, _ = generator(latent, input_is_latent=True, stns=stns, rt_stns=rt_stns, alpha=args.alpha)
torchvision.utils.save_image(result, f'{output_dir}/{args.style}_{os.path.basename(args.input_image)}',
                             normalize=True, value_range=(-1, 1))
