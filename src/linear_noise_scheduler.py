import torch


class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end) -> None:
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas

        # Precompute for immediate use
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
    
    def add_noise(self, original, noise, t):
        """
        Forward process: add noise to the original sample.
        """
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)

        # Ensure input shape matches
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        return (sqrt_alpha_cum_prod.to(original.device) * original + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)

    def sample_prev_timestep(self, xt, noise_pred, t):
        """
        Backward process: sample previous timestep.
        """
        batch_size = xt.shape[0]
        
        sqrt_one_minus_alpha_cum_prod_t = self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t].view(batch_size, 1, 1, 1)
        alpha_cum_prod_t = self.alpha_cum_prod.to(xt.device)[t].view(batch_size, 1, 1, 1)
        betas_t = self.betas.to(xt.device)[t].view(batch_size, 1, 1, 1)
        alphas_t = self.alphas.to(xt.device)[t].view(batch_size, 1, 1, 1)

        x0 = (xt - (sqrt_one_minus_alpha_cum_prod_t * noise_pred)) / torch.sqrt(alpha_cum_prod_t)
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - (betas_t * noise_pred) / sqrt_one_minus_alpha_cum_prod_t
        mean = mean / torch.sqrt(alphas_t)

        if torch.all(t == 0):
            return mean, x0
        
        alpha_cum_prod_prev = self.alpha_cum_prod.to(xt.device)[torch.clamp(t-1, min=0)].view(batch_size, 1, 1, 1)
        variance = (1 - alpha_cum_prod_prev) / (1.0 - alpha_cum_prod_t)
        variance = variance * betas_t
        
        sigma = variance ** 0.5
        z = torch.randn_like(xt)
        
        # Disable noise when t == 0
        nonzero_mask = (t != 0).float().view(batch_size, 1, 1, 1).to(xt.device)

        return mean + nonzero_mask * sigma * z, x0