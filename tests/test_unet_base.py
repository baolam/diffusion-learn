import torch
import unittest
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.unet_base import UNet, DownBlock, MidBlock, UpBlock

class TestUNet(unittest.TestCase):
    def setUp(self):
        self.model_configs = {
            "im_channels": 3,
            "down_channels": [32, 64, 128],
            "mid_channels": [128, 64],
            "time_emb_dim": 128,
            "down_sample": [True, True],
            "num_down_layers": 2,
            "num_mid_layers": 2,
            "num_up_layers": 2
        }

    def test_unet_instantiation_and_forward(self):
        model = UNet(self.model_configs)
        self.assertIsInstance(model, UNet)

        # Create dummy input
        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([10, 50])

        # Forward pass
        out = model(x, t)

        # Check output shape
        self.assertEqual(out.shape, (2, 3, 32, 32))

if __name__ == '__main__':
    unittest.main()
