import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from data_manager.mnist_dataset import get_dataset
from torch.utils.data import DataLoader
from src.unet_base import UNet
from src.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    with open(args.config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    
    print(config)

    ### Get config ###
    diffusion_config = config["diffusion_params"]
    dataset_config = config["dataset_params"]
    model_config = config["model_params"]
    train_config = config["train_params"]

    scheduler = LinearNoiseScheduler(**diffusion_config)

    mnist = get_dataset(im_path=dataset_config['im_path'])
    mnist_loader = DataLoader(mnist, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)

    model = UNet(model_config).to(device)
    model.train()

    # if not os.path.exists(train_config['task_name']):
    #     os.makedirs(train_config[])

    if os.path.exists(os.path.join(train_config['ckpt_name'])):
        print("Loading checkpoint as found one!")
        model.load_state_dict(torch.load(train_config['ckpt_name'], map_location=device))

    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    for epoch_idx in range(num_epochs):
        losses = []

        for im, _ in tqdm(mnist_loader):
            optimizer.zero_grad()

            im = im.float().to(device)

            noise = torch.rand_like(im).to(device)

            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0], )).to(device)

            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noisy_im)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
        
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses),
        ))

        torch.save(model.state_dict(), os.path.join(train_config['ckpt_name']))
        
        if args.drive != "No":
            # Path to store in drive!
            # /content/drive/MyDrive/checkpoints/
            torch.save(model.state_dict(), os.path.join(args.drive, "ddpm_ckpt.pth"))
            print("Stored to drive!")
            
        print("Done training!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm training")
    parser.add_argument('--config', dest='config_path', default='configs/default.yaml', type=str)
    parser.add_argument('--drive', dest='strorage drive', type=str, default="No")

    args = parser.parse_args()
    train(args)