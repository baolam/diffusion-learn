import torch


class DDIM_NoiseScheduler:
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
    

    def sample_prev_timestep(self, xt, noise_pred, t, prev_t, eta=0.):
        alpha_cum_prod_t = self.alpha_cum_prod[t].view(-1, 1, 1, 1).to(xt.device)

        if prev_t >= 0:
            alpha_cum_prod_prev = self.alpha_cum_prod[prev_t].view(-1, 1, 1, 1).to(xt.device)
        else:
            alpha_cum_prod_prev = torch.ones_like(alpha_cum_prod_t)
        
        pred_x0 = (xt - torch.sqrt(1 - alpha_cum_prod_t) * noise_pred) / torch.sqrt(alpha_cum_prod_t)

        pred_x0 = torch.clamp(pred_x0, -1., 1.)

        if eta > 0:
            sigma = eta * torch.sqrt((1 - alpha_cum_prod_prev) / (1 - alpha_cum_prod_t)) * torch.sqrt(1 - alpha_cum_prod_t / alpha_cum_prod_prev)
        else:
            sigma = 0.0

        direction_xt = torch.sqrt(1 - alpha_cum_prod_prev - sigma**2) * noise_pred

        prev_xt = torch.sqrt(alpha_cum_prod_prev) * pred_x0 + direction_xt

        if eta > 0 and prev_t >= 0:
            noise = torch.randn_like(xt)
            prev_xt += sigma * noise

        return prev_xt