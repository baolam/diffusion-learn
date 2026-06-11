import unittest
import torch
import yaml
from src.unet_base import UNet


class TestUNet(unittest.TestCase):
    def _load_model_config(self):
        with open("configs/default.yaml", 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
        
        return config["model_params"]
    
    def setUp(self) -> None:
        self.model = UNet(self._load_model_config())
        self.sample_input_x = torch.randn(3, 1, 28, 28)
        self.sample_input_t = [3, 4, 7]
        
        # Thử thông tin về số lượng tham số
        totals = sum(p.numel() for p in self.model.parameters())
        # Cỡ 10M tham số :))
        print("Total parameters:", totals)

        
    def test_output_shape(self):
        output = self.model(self.sample_input_x, self.sample_input_t)

        self.assertEqual(output.shape, (3, 1, 28, 28), msg=f"Shape mismatch! Need (3, 1, 28, 28) but received {list(output.shape)}")

    def test_convergence(self):
        pass


if __name__ == "__main__":
    unittest.main()