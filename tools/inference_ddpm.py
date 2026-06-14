import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from src.unet_base import UNet
from src.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model : UNet, scheduler : LinearNoiseScheduler, train_config, model_config, diffusion_config):
    xt = torch.randn((train_config['num_samples'], model_config['im_channels'], model_config['im_size'], model_config['im_size'])).to(device)

    print("Sample image:", xt.shape)

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save x0
        ims = torch.clamp(x0_pred, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        # ims = torch.clamp(ims, -1., 1.).detach().cpu()

        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)

        img.save(os.path.join('samples', "mnist_x0_{}.png".format(i)))
        img.close()

def infer(args):
    with open(args.config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    
    print(config)

    diffusion_config = config["diffusion_params"]
    model_config = config["model_params"]
    train_config = config["train_params"]

    model = UNet(model_config).to(device)
    model.load_state_dict(torch.load(train_config['ckpt_name'], map_location=device))
    model.eval()
    print("Load model completed!")

    scheduler = LinearNoiseScheduler(**diffusion_config)

    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='configs/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)
