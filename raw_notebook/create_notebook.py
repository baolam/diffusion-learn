import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# The 4 Steps of Diffusion Models\n",
    "\n",
    "This notebook illustrates the diffusion process broken down into 4 conceptual steps based on the simple-to-complex perspective.\n",
    "\n",
    "1. **Step 1**: Choose an easy-to-sample distribution (usually Gaussian), $p(X_T)$\n",
    "2. **Step 2**: Parameterize the conditional distribution $p(X_{t-1}|X_t)$ or $p(X_{t-1})$\n",
    "3. **Step 3**: Sample $X_{t-1}$ following the parameterized distribution\n",
    "4. **Step 4**: Repeat the process until $t = 0$"
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
    "## Setup: Create a simple 2D Dataset and define forward diffusion\n",
    "Before generating, we need data and a forward process to train our parameterization."
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
    "# --- Diffusion Hyperparameters ---\n",
    "n_steps = 100\n",
    "beta = torch.linspace(0.0001, 0.02, n_steps)\n",
    "alpha = 1. - beta\n",
    "alpha_bar = torch.cumprod(alpha, dim=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Parameterize $p(X_{t-1} | X_t)$\n",
    "We use a simple Neural Network to predict the noise added at step $t$, which is equivalent to parameterizing $p(X_{t-1} | X_t)$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class SimpleDiffusionModel(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.net = nn.Sequential(\n",
    "            nn.Linear(2 + 1, 64), # 2D data + 1D time\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(64, 64),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(64, 2)\n",
    "        )\n",
    "        \n",
    "    def forward(self, x, t):\n",
    "        # Normalize time\n",
    "        t = t.unsqueeze(-1).float() / n_steps\n",
    "        x_t = torch.cat([x, t], dim=-1)\n",
    "        return self.net(x_t)\n",
    "\n",
    "model = SimpleDiffusionModel()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n",
    "\n",
    "print(\"Training the parameterization model (this takes a few seconds)...\")\n",
    "epochs = 1000\n",
    "batch_size = 128\n",
    "for epoch in range(epochs):\n",
    "    idx = torch.randperm(X.shape[0])[:batch_size]\n",
    "    x0 = X[idx]\n",
    "    \n",
    "    t = torch.randint(0, n_steps, (batch_size,))\n",
    "    noise = torch.randn_like(x0)\n",
    "    \n",
    "    a_bar_t = alpha_bar[t].unsqueeze(-1)\n",
    "    xt = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1 - a_bar_t) * noise\n",
    "    \n",
    "    pred_noise = model(xt, t)\n",
    "    loss = nn.MSELoss()(pred_noise, noise)\n",
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
    "Here, we start with a standard Gaussian distribution."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_samples = 1000\n",
    "x_T = torch.randn(n_samples, 2)\n",
    "x_t = x_T.clone()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3 & 4: Sample $X_{t-1}$ and Repeat until $t=0$\n",
    "We iteratively apply our parameterized reverse process to denoise the samples."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.eval()\n",
    "with torch.no_grad():\n",
    "    # Step 4: Repeat until t = 0\n",
    "    for i in reversed(range(n_steps)):\n",
    "        t_tensor = torch.full((n_samples,), i, dtype=torch.long)\n",
    "        \n",
    "        # Predict the noise using our parameterized model\n",
    "        pred_noise = model(x_t, t_tensor)\n",
    "        \n",
    "        alpha_t = alpha[i]\n",
    "        alpha_bar_t = alpha_bar[i]\n",
    "        \n",
    "        # Step 3: Sample x_{t-1}\n",
    "        if i > 0:\n",
    "            noise = torch.randn_like(x_t)\n",
    "        else:\n",
    "            noise = torch.zeros_like(x_t)\n",
    "            \n",
    "        x_t = (1 / torch.sqrt(alpha_t)) * (\n",
    "            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise\n",
    "        ) + torch.sqrt(beta[i]) * noise\n",
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
    "plt.title(\"Generated Data (After 4 Steps)\")\n",
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
with open(r"e:\simulate-papers\Simple-Diffusion\notebooks\diffusion_4_steps.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)
