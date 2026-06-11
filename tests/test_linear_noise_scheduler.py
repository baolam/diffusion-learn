import torch
import unittest
import yaml

from src.linear_noise_scheduler import LinearNoiseScheduler


class TestLinearNoiseScheduler(unittest.TestCase):
    def _load_diffusion_params(self):
        with open("configs/default.yaml", 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
        
        return config["diffusion_params"]

    def setUp(self) -> None:
        params = self._load_diffusion_params()

        self.num_timesteps = params["num_timesteps"]
        self.scheduler = LinearNoiseScheduler(**params)

        self.batch_size = 2
        self.img_shape = (self.batch_size, 1, 28, 28)
        self.original_img = torch.randn(self.img_shape)
        self.noise = torch.randn(self.img_shape)
    
    def test_math_logic(self):
        self.assertTrue(torch.all(self.scheduler.betas[1:] >= self.scheduler.betas[:-1]))

        self.assertTrue(torch.all(self.scheduler.alpha_cum_prod[1:] <= self.scheduler.alpha_cum_prod[:-1]))
        self.assertTrue(torch.all(self.scheduler.alpha_cum_prod > 0))
        self.assertTrue(torch.all(self.scheduler.alpha_cum_prod <= 1.0))


    def test_add_noise_shape_and_boundary(self):
        t = torch.randint(0, self.num_timesteps, (self.batch_size,))
    
        noisy_img = self.scheduler.add_noise(self.original_img, self.noise, t)
        
        self.assertEqual(noisy_img.shape, self.original_img.shape)
        
        # t_zeros = torch.zeros(self.batch_size, dtype=torch.long)
        # noisy_at_zero = self.scheduler.add_noise(self.original_img, self.noise, t_zeros)
        
        # self.assertTrue(torch.allclose(noisy_at_zero, self.original_img, atol=1e-2))

    # def test_sample_prevtimestep_boundary_x0(self):
    #     xt = torch.randn(self.img_shape)
    #     noise_pred = torch.randn(self.img_shape)
    #     t_zero = torch.tensor([0, 0], dtype=torch.long)
        
    #     try:
    #         pred_prev, x0 = self.scheduler.sample_prev_timestep(xt, noise_pred, t_zero)
    #     except Exception as e:
    #         self.fail(f"sample_prev_timestep raised error at t=0: {e}")
            
    #     self.assertEqual(pred_prev.shape, self.img_shape)
    #     self.assertEqual(x0.shape, self.img_shape)
        
    #     self.assertTrue(torch.all(x0 >= -1.0) and torch.all(x0 <= 1.0))
    
    # def test_sample_prev_timestep_general(self):
    #     xt = torch.randn(self.img_shape)
    #     noise_pred = torch.randn(self.img_shape)
    #     t_general = torch.tensor([50, 50], dtype=torch.long) # t > 0
        
    #     pred_prev, x0 = self.scheduler.sample_prev_timestep(xt, noise_pred, t_general)
        
    #     self.assertEqual(pred_prev.shape, self.img_shape)
    #     self.assertEqual(x0.shape, self.img_shape)

    def test_device_compatibility(self):
        if not torch.cuda.is_available():
            self.skipTest("No GPU for testing")
            
        device = torch.device("cuda")
        
        gpu_img = self.original_img.to(device)
        gpu_noise = self.noise.to(device)
        t = torch.tensor([10, 10], dtype=torch.long).to(device)
        
        try:
            gpu_output = self.scheduler.add_noise(gpu_img, gpu_noise, t)
            self.assertEqual(gpu_output.device.type, "cuda")
        except Exception as e:
            self.fail(f"add_noise crashed when running on GPU: {e}")

        try:
            pred_prev, x0 = self.scheduler.sample_prev_timestep(gpu_img, gpu_noise, t)
            self.assertEqual(pred_prev.device.type, "cuda")
            self.assertEqual(x0.device.type, "cuda")
        except Exception as e:
            self.fail(f"sample_prev_timestep crashed when running on GPU: {e}")