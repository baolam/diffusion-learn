import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Flow Matching\n",
    "\n",
    "This notebook illustrates Flow Matching on a simple 2D dataset.\n",
    "\n",
    "Flow Matching is a framework for training continuous normalizing flows using vector fields. The basic idea is to construct a simple path between a base distribution (e.g., standard Gaussian) and the target data distribution, and then train a neural network to match the vector field of this path.\n",
    "\n",
    "1. **Path Construction**: Define $x_t = (1-t) x_0 + t x_1$, where $x_0 \sim \mathcal{N}(0, I)$ and $x_1 \sim p_{data}(x)$ for $t \in [0, 1]$.\n",
    "2. **Vector Field**: The target velocity is $u_t(x_t) = x_1 - x_0$.\n",
    "3. **Training**: Train a neural network $v_\\theta(x, t)$ to minimize $\mathbb{E}_{t, x_0, x_1}[||v_\\theta(x_t, t) - u_t(x_t)||^2]$.\n",
    "4. **Sampling**: Solve the ODE $dx = v_\\theta(x, t)dt$ from $t=0$ to $t=1$ starting from $x_0 \sim \mathcal{N}(0, I)$."
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
    "## Setup: Create a simple 2D Dataset\n",
    "We use the moons dataset as our target distribution $p_{data}$."
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
    "X = torch.tensor(X, dtype=torch.float32)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Network Architecture\n",
    "We use a simple MLP to predict the velocity $v_\\theta(x, t)$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class VelocityNet(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.net = nn.Sequential(\n",
    "            nn.Linear(2 + 1, 128),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(128, 128),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(128, 128),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(128, 2)\n",
    "        )\n",
    "        \n",
    "    def forward(self, x, t):\n",
    "        # t should be shape (batch_size, 1)\n",
    "        x_t = torch.cat([x, t], dim=-1)\n",
    "        return self.net(x_t)\n",
    "\n",
    "model = VelocityNet()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Training\n",
    "We sample $t \sim \mathcal{U}(0, 1)$, $x_0 \sim \mathcal{N}(0, I)$, and $x_1 \sim p_{data}$. Then we compute $x_t$ and regress the network $v_\\theta(x_t, t)$ to $x_1 - x_0$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Training the velocity network...\")\n",
    "epochs = 2000\n",
    "batch_size = 256\n",
    "for epoch in range(epochs):\n",
    "    # Sample target data (x_1)\n",
    "    idx = torch.randperm(X.shape[0])[:batch_size]\n",
    "    x1 = X[idx]\n",
    "    \n",
    "    # Sample base data (x_0)\n",
    "    x0 = torch.randn_like(x1)\n",
    "    \n",
    "    # Sample time t\n",
    "    t = torch.rand(x1.shape[0], 1)\n",
    "    \n",
    "    # Interpolate to get x_t\n",
    "    xt = (1 - t) * x0 + t * x1\n",
    "    \n",
    "    # Target velocity\n",
    "    ut = x1 - x0\n",
    "    \n",
    "    # Predict velocity\n",
    "    vt = model(xt, t)\n",
    "    \n",
    "    # Flow Matching Loss\n",
    "    loss = torch.mean((vt - ut)**2)\n",
    "    \n",
    "    optimizer.zero_grad()\n",
    "    loss.backward()\n",
    "    optimizer.step()\n",
    "    \n",
    "    if (epoch + 1) % 500 == 0:\n",
    "        print(f\"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}\")\n",
    "\n",
    "print(\"Training complete!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Sampling (Inference)\n",
    "To generate data, we sample $x_0 \sim \mathcal{N}(0, I)$ and solve the ODE $dx = v_\\theta(x, t) dt$ from $t=0$ to $t=1$ using a simple Euler method."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.eval()\n",
    "n_samples = 1000\n",
    "x_0 = torch.randn(n_samples, 2)\n",
    "x_t = x_0.clone()\n",
    "\n",
    "n_steps = 100\n",
    "dt = 1.0 / n_steps\n",
    "\n",
    "with torch.no_grad():\n",
    "    for i in range(n_steps):\n",
    "        t = torch.full((n_samples, 1), i * dt)\n",
    "        v = model(x_t, t)\n",
    "        x_t = x_t + v * dt\n",
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
    "plt.scatter(x_t[:, 0], x_t[:, 1], alpha=0.5, s=5, color='green')\n",
    "plt.title(\"Generated Data (Flow Matching)\")\n",
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
with open(r"e:\simulate-papers\Simple-Diffusion\notebooks\flow_matching.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)
