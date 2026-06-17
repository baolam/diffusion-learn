import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from src.unet_base import UNet
from src.ddim_noise_scheduler import DDIM_NoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model : UNet, scheduler : DDIM_NoiseScheduler, train_config, model_config, diffusion_config):
    xt = torch.randn((train_config['num_samples'], model_config['im_channels'], model_config['im_size'], model_config['im_size'])).to(device)

    times = torch.linspace(diffusion_config["num_timesteps"] - 1, 0, 50, dtype=torch.long).tolist()

    for i in tqdm(range(len(times))):
        t_curr = times[i]
        t_prev = times[i+1] if i < len(times) - 1 else -1

        t_tensor = torch.full((train_config['num_samples'],), t_curr, dtype=torch.long).to(device)

        noise_pred = model(xt, t_tensor)
        xt = scheduler.sample_prev_timestep(
            xt=xt, 
            noise_pred=noise_pred, 
            t=t_tensor, 
            prev_t=t_prev
        )

        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2

        grid = make_grid(ims, nrow=len(times))
        img = torchvision.transforms.ToPILImage()(grid)

        img.save(os.path.join('samples', 'mnist_x0_{}.png'.format(i)))
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

    scheduler = DDIM_NoiseScheduler(**diffusion_config)

    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='configs/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)
