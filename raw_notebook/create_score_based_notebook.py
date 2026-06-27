import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# The 4 Steps of Score-Based Generative Models\n",
    "\n",
    "This notebook illustrates the score-based generative modeling process (like NCSN) broken down into the same 4 conceptual steps.\n",
    "\n",
    "1. **Step 1**: Choose an easy-to-sample distribution (usually Gaussian with large variance), $p(X_T)$\n",
    "2. **Step 2**: Parameterize the conditional distribution / score function $\\nabla_x \\log p(x_t)$ using a neural network\n",
    "3. **Step 3**: Sample $X_{t-1}$ by moving along the score direction with added noise (Langevin Dynamics)\n",
    "4. **Step 4**: Repeat the process until $t = 0$ (Annealed Langevin Dynamics)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from sklearn.datasets import make_moons\n",
    "\n",
    "# Set random seed for reproducibility\n",
    "torch.manual_seed(42)\n",
    "np.random.seed(42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setup: Create a simple 2D Dataset and Noise Schedule\n",
    "We use a geometric progression of noise scales $\\sigma_1 < \\sigma_2 < \\dots < \\sigma_L$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Data Preparation ---\n",
    "X, _ = make_moons(n_samples=2000, noise=0.05)\n",
    "X = torch.tensor(X, dtype=torch.float32)\n",
    "\n",
    "# --- Noise Schedule (Variance Exploding style) ---\n",
    "n_steps = 100\n",
    "sigma_min = 0.01\n",
    "sigma_max = 2.0\n",
    "sigmas = torch.exp(torch.linspace(np.log(sigma_max), np.log(sigma_min), n_steps))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Parameterize the Score Function $\\nabla_{x} \\log p(x_t)$\n",
    "We use a Neural Network to predict the score. We train it using Denoising Score Matching.\n",
    "The target score for $p(x_t | x_0)$ is $-(x_t - x_0) / \\sigma_t^2$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class ScoreNet(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.net = nn.Sequential(\n",
    "            nn.Linear(2 + 1, 64),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(64, 64),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(64, 2)\n",
    "        )\n",
    "        \n",
    "    def forward(self, x, t_idx):\n",
    "        # Normalize time index\n",
    "        t_norm = t_idx.unsqueeze(-1).float() / n_steps\n",
    "        x_t = torch.cat([x, t_norm], dim=-1)\n",
    "        # To help the network scale, we divide the output by sigma\n",
    "        # as the score magnitude is proportional to 1/sigma\n",
    "        return self.net(x_t) / sigmas[t_idx].unsqueeze(-1)\n",
    "\n",
    "model = ScoreNet()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n",
    "\n",
    "print(\"Training the score network (this takes a few seconds)...\")\n",
    "epochs = 1000\n",
    "batch_size = 128\n",
    "for epoch in range(epochs):\n",
    "    idx = torch.randperm(X.shape[0])[:batch_size]\n",
    "    x0 = X[idx]\n",
    "    \n",
    "    # Sample random noise levels\n",
    "    t_idx = torch.randint(0, n_steps, (batch_size,))\n",
    "    used_sigmas = sigmas[t_idx].unsqueeze(-1)\n",
    "    \n",
    "    # Add noise to data\n",
    "    noise = torch.randn_like(x0)\n",
    "    xt = x0 + noise * used_sigmas\n",
    "    \n",
    "    # Denoising Score Matching objective\n",
    "    target_score = -noise / used_sigmas\n",
    "    pred_score = model(xt, t_idx)\n",
    "    \n",
    "    # Loss weighted by sigma^2 to balance magnitude across noise scales\n",
    "    loss = torch.mean(used_sigmas**2 * (pred_score - target_score)**2)\n",
    "    \n",
    "    optimizer.zero_grad()\n",
    "    loss.backward()\n",
    "    optimizer.step()\n",
    "print(\"Training complete!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1: Choose easy-to-sample distribution $p(X_T)$\n",
    "We start with a Gaussian distribution with variance $\\sigma_{max}^2$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_samples = 1000\n",
    "x_T = torch.randn(n_samples, 2) * sigma_max\n",
    "x_t = x_T.clone()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3 & 4: Sample $X_{t-1}$ and Repeat until $t=0$\n",
    "We use Annealed Langevin Dynamics to sample from the sequence of distributions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.eval()\n",
    "n_langevin_steps = 10 # Steps per noise scale\n",
    "eps = 2e-5 # Step size coefficient\n",
    "\n",
    "with torch.no_grad():\n",
    "    # Step 4: Repeat for all noise scales from max to min\n",
    "    for i in range(n_steps):\n",
    "        t_tensor = torch.full((n_samples,), i, dtype=torch.long)\n",
    "        current_sigma = sigmas[i]\n",
    "        \n",
    "        # Calculate step size alpha_i for Langevin Dynamics\n",
    "        step_size = eps * (current_sigma / sigmas[-1]) ** 2\n",
    "        \n",
    "        # Step 3: Langevin dynamics steps at current noise scale\n",
    "        for _ in range(n_langevin_steps):\n",
    "            score = model(x_t, t_tensor)\n",
    "            noise = torch.randn_like(x_t)\n",
    "            \n",
    "            # Update step\n",
    "            x_t = x_t + 0.5 * step_size * score + torch.sqrt(step_size) * noise\n",
    "\n",
    "# Visualize the results\n",
    "plt.figure(figsize=(12, 5))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.scatter(X[:, 0], X[:, 1], alpha=0.5, s=5)\n",
    "plt.title(\"Original Data Distribution (Target)\")\n",
    "plt.xlim(-1.5, 2.5)\n",
    "plt.ylim(-1.0, 1.5)\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.scatter(x_t[:, 0], x_t[:, 1], alpha=0.5, s=5, color='orange')\n",
    "plt.title(\"Generated Data (Score-Based)\")\n",
    "plt.xlim(-1.5, 2.5)\n",
    "plt.ylim(-1.0, 1.5)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

os.makedirs(r"e:\simulate-papers\Simple-Diffusion\notebooks", exist_ok=True)
with open(r"e:\simulate-papers\Simple-Diffusion\notebooks\score_based_4_steps.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)
