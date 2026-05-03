#! /usr/bin/env python3

import torch
from torch import nn
import matplotlib.pyplot as plt

# ─── TARGET FUNCTION ──────────────────────────────────────────────────────────
# A mix of oscillation, a Gaussian bump, and a polynomial trend on [-5, 5].
# - sin(2x): fast oscillation
# - cos(x/2): slow modulation of the envelope
# - exp(-0.2 * x^2): Gaussian that suppresses signal away from centre
# - 0.05 * x^3: gentle asymmetric trend
# This combination has no closed-form easy structure, making it genuinely hard.
def target(x):
    return torch.sin(2 * x) * torch.cos(x / 2) + torch.exp(-0.2 * x**2) + 0.05 * x**3

# ─── DATASET ──────────────────────────────────────────────────────────────────
# 200 evenly spaced x values; unsqueeze(-1) turns shape [200] → [200, 1]
# because nn.Linear expects (batch_size, features).
x_train = torch.linspace(-5, 5, 400).unsqueeze(-1)
y_train = target(x_train)

# ─── MODEL ────────────────────────────────────────────────────────────────────
# Subclassing nn.Module lets us define the forward pass explicitly, which is
# necessary for anything beyond a simple chain (skip connections, branching, etc.).
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)   # 1 input feature → 64 hidden units
        self.fc2 = nn.Linear(64, 64)  # 64 → 64 (second hidden layer adds expressivity)
        self.fc3 = nn.Linear(64, 1)   # 64 hidden units → 1 output value

    # forward() is called when you do model(x) — PyTorch routes it here automatically.
    # Tanh is smooth and works well for function approximation (unlike ReLU which is piecewise linear).
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

model = MLP()

# ─── LOSS ─────────────────────────────────────────────────────────────────────
# MSE = mean over batch of (prediction - target)^2.
# This is the standard loss for regression tasks.
loss_fn = nn.MSELoss()

# ─── OPTIMIZER ────────────────────────────────────────────────────────────────
# Adam adjusts each parameter's learning rate based on gradient history.
# lr=1e-3 is a safe default; too large → diverges, too small → slow.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ─── TRAINING LOOP ────────────────────────────────────────────────────────────
# One "epoch" = one full pass over the entire training set.
losses = []

for epoch in range(5000):
    # 1. Forward pass: compute predictions
    y_pred = model(x_train)

    # 2. Compute loss
    loss = loss_fn(y_pred, y_train)

    # 3. Zero gradients — PyTorch accumulates them by default, so clear each step
    optimizer.zero_grad()

    # 4. Backward pass: compute d(loss)/d(every parameter) via chain rule
    loss.backward()

    # 5. Gradient step: move each parameter in the direction that reduces loss
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1:4d} | Loss: {loss.item():.6f}")

# ─── PLOT ─────────────────────────────────────────────────────────────────────
x_test = torch.linspace(-5, 5, 500).unsqueeze(-1)

# torch.no_grad() disables gradient tracking — we don't need it for inference
with torch.no_grad():
    y_pred_final = model(x_test)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x_test, target(x_test), label="Target", linewidth=2)
ax1.plot(x_test, y_pred_final, label="Network", linestyle="--", linewidth=2)
ax1.set_title("Function approximation")
ax1.legend()

ax2.semilogy(losses)
ax2.set_title("Training loss (log scale)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE")

plt.tight_layout()
plt.savefig("fit.png", dpi=150)
plt.show()
print("Saved fit.png")
