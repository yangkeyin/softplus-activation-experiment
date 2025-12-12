import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legfit

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Data Generation
# ==========================================
# We need a dense grid for accurate numerical integration/fitting of Legendre coefficients later.
N_points = 4000 
x_np = np.linspace(-1, 1, N_points)

# Target Function: Sum of sines with distinct frequencies
# Despite having equal amplitude in Fourier basis (sines), 
# they will have a decaying spectrum in Legendre basis.
frequencies = [5, 15, 30, 50, 80]
y_np = np.zeros_like(x_np)
for f in frequencies:
    y_np += np.sin(f * np.pi * x_np)

# Normalize target to roughly [-1, 1] range for stable training
y_np = y_np / len(frequencies)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(x_np).reshape(-1, 1)
Y_train = torch.FloatTensor(y_np).reshape(-1, 1)

# ==========================================
# 2. Model Definition (MLP with Tanh)
# ==========================================
class MLP(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        # Tanh activation is crucial for smooth function approximation 
        # and clear spectral analysis free of ReLU artifacts.
        self.net = nn.Sequential(
            nn.Linear(1, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP().cuda() if torch.cuda.is_available() else MLP()
X_train = X_train.cuda() if torch.cuda.is_available() else X_train
Y_train = Y_train.cuda() if torch.cuda.is_available() else Y_train

# ==========================================
# 3. Training Loop with Checkpoints
# ==========================================
optimizer = optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.MSELoss()

# Checkpoints to capture the evolution from early to late training
checkpoints = [200, 1000, 5000, 15000, 30000]
max_epochs = checkpoints[-1]
saved_predictions = {}

print(f"Training on device: {X_train.device}")
print(f"Target frequencies: {frequencies}")

for epoch in range(1, max_epochs + 1):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()

    if epoch in checkpoints or epoch == 1:
        print(f"Epoch {epoch:6d} | Loss: {loss.item():.8f}")
        # Save model prediction on CPU numpy array for analysis
        saved_predictions[epoch] = output.detach().cpu().numpy().flatten()

# ==========================================
# 4. Spectral Analysis (Legendre Coefficients)
# ==========================================
def get_legendre_coeffs(x_grid, y_data, max_degree):
    """
    Computes Legendre coefficients using least squares fitting (legfit).
    Returns the absolute magnitude of coefficients up to max_degree.
    """
    # legfit returns coefficients [c0, c1, c2, ..., c_n]
    coeffs = legfit(x_grid, y_data, max_degree)
    return np.abs(coeffs)

# Max degree to analyze. Needs to be high enough to capture the highest frequency component.
# Frequency f roughly corresponds to Legendre degree n ~= pi * f
max_analyze_degree = int(np.pi * max(frequencies) * 1.2) 
degrees = np.arange(max_analyze_degree + 1)

# Calculate coefficients for the Ground Truth Target
target_coeffs = get_legendre_coeffs(x_np, y_np, max_analyze_degree)

# ==========================================
# 5. Visualization
# ==========================================
plt.figure(figsize=(14, 8))

# Plot Target Spectrum (Black line)
plt.scatter(degrees, target_coeffs, s=15, marker='o', color='k', zorder=10, label='Target Function Spectrum')

# Prepare colors for checkpoints (using acolormap for time progression)
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 0.9, len(checkpoints))]

# Plot Model Predictions at Checkpoints
for i, epoch in enumerate(checkpoints):
    pred_y = saved_predictions[epoch]
    pred_coeffs = get_legendre_coeffs(x_np, pred_y, max_analyze_degree)
    
    plt.plot(degrees, pred_coeffs, 
             color=colors[i], linewidth=1.5, alpha = 0.8, linestyle='-', 
             label=f'Epoch {epoch}')

# Finalize plot styling
plt.yscale('log')
plt.xlim(0, max_analyze_degree)
# Set reasonable y-limits to see the noise floor vs signal
plt.ylim(1e-6, np.max(target_coeffs)*1.5) 

plt.title("Visualizing Spectral Bias: Legendre Coefficient Evolution over Training", fontsize=16)
plt.xlabel("Legendre Polynomial Degree ($n$)\n(Roughly proportional to Frequency)", fontsize=14)
plt.ylabel("Coefficient Amplitude $|c_n|$ (Log Scale)", fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.legend(fontsize=12)

# Add annotation arrows to emphasize the phenomenon
plt.annotate('Low degrees fit first', xy=(10, 1e-2), xytext=(30, 1e-1),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)
plt.annotate('High degrees fit later', xy=(150, 1e-4), xytext=(80, 1e-3),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

plt.tight_layout()

print("Training complete. Generating plot...")
plt.show()