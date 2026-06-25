import torch
import unittest
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.linear_noise_scheduler import LinearNoiseScheduler
from src.ddim_noise_scheduler import DDIM_NoiseScheduler

class TestNoiseSchedulers(unittest.TestCase):
    def test_linear_noise_scheduler(self):
        scheduler = LinearNoiseScheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
        self.assertEqual(scheduler.betas.shape, (1000,))
        
        # Test add_noise
        original = torch.randn(4, 3, 32, 32)
        noise = torch.randn(4, 3, 32, 32)
        t = torch.tensor([100, 100, 100, 100])
        noisy_x = scheduler.add_noise(original, noise, t)
        self.assertEqual(noisy_x.shape, (4, 3, 32, 32))

        # Test sample_prev_timestep
        noise_pred = torch.randn(4, 3, 32, 32)
        mean, x0 = scheduler.sample_prev_timestep(noisy_x, noise_pred, t)
        self.assertEqual(mean.shape, (4, 3, 32, 32))
        self.assertEqual(x0.shape, (4, 3, 32, 32))

    def test_ddim_noise_scheduler(self):
        scheduler = DDIM_NoiseScheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
        self.assertEqual(scheduler.betas.shape, (1000,))
        
        # Test sample_prev_timestep
        xt = torch.randn(4, 3, 32, 32)
        noise_pred = torch.randn(4, 3, 32, 32)
        t = 100
        prev_t = 80
        
        prev_xt = scheduler.sample_prev_timestep(xt, noise_pred, t, prev_t)
        self.assertEqual(prev_xt.shape, (4, 3, 32, 32))

if __name__ == '__main__':
    unittest.main()
