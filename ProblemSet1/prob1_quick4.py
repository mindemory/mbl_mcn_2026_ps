"""
Quick answers for Problem 1 — Parts 1.1, 1.1.1, 1.2.1, 1.2.2
Run from the repo root:  python ProblemSet1/prob1_quick4.py
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('dark_background')

# ── Load data ──────────────────────────────────────────────────────────────
data = loadmat(os.path.join(os.getcwd(), 'ProblemSet1', 'HatsopoulosReachTask.mat'))

numNeurons       = data['numNeurons'].item()
numTimebins      = data['numTimebins'].item()
uniqueDirections = np.unique(data['direction'].T)
numDirs          = len(uniqueDirections)

# ── 1.1  Build 3-D trial-averaged array & flatten to 2-D ──────────────────
trial_avg = np.empty((numNeurons, numTimebins, numDirs))
for idir in uniqueDirections:
    idx = np.where(data['direction'].T == idir)[1]
    trial_avg[:, :, idir - 1] = np.mean(data['firingRate'][:, :, idx], axis=2)

# Transpose to (numNeurons, numDirs, numTimebins) so directions are the outer
# (slower) axis — i.e. all timebins for dir1, then all for dir2, etc.
X = trial_avg.transpose(0, 2, 1).reshape(numNeurons, numDirs * numTimebins)

print("=" * 60)
print("1.1  3-D trial-averaged array shape:", trial_avg.shape)
print(f"     (numNeurons x numTimebins x numDirs) = "
      f"({numNeurons} x {numTimebins} x {numDirs})")
print()
print("1.1  2-D data matrix X shape:", X.shape)
print(f"     (numNeurons x numTimebins*numDirs) = "
      f"({numNeurons} x {numTimebins*numDirs})")

# ── 1.1.1  Goal of averaging & concatenating ──────────────────────────────
print()
print("=" * 60)
print("1.1.1  What is the goal of averaging and concatenating?")
print()
print("  * Trial-averaging: reduce noise by averaging repeated trials of the")
print("    same reach direction, leaving the direction-tuned signal.")
print()
print("  * Concatenation along directions: stacking the 8 direction blocks")
print("    side-by-side gives a single 2-D matrix where each ROW is one")
print("    neuron's full time-course across all directions.")
print()
print("  * This form is required for PCA/SVD, which decompose 2-D matrices.")
print("    The structure captured is: how much of the neural population")
print("    variance is shared, and which 'directions' in neuron space best")
print("    explain the reach-direction tuning over time.")

# ── 1.2.1  Row mean-centre & display ──────────────────────────────────────
X_c = X - X.mean(axis=1, keepdims=True)

print()
print("=" * 60)
print("1.2.1  Mean-centered data matrix X_c shape:", X_c.shape)
print()
print("  * Each row (neuron) has its mean firing rate subtracted.")
print("  * Removes the baseline / tonic firing rate of each neuron.")
print("  * Remaining signal = direction- and time-modulated activity.")
print("  * Structure you should see in the heatmap:")
print("    - Red/blue stripes across columns (directions) -- some neurons")
print("      prefer certain directions (direction tuning).")
print("    - Temporal structure within each direction block.")
print("    - Most neurons show ~same sign for the same direction block,")
print("      reflecting population-level tuning similarity.")

# ── 1.2.2  Covariance matrix ───────────────────────────────────────────────
T   = X_c.shape[1]
C   = (X_c @ X_c.T) / (T - 1)

print()
print("=" * 60)
print("1.2.2  Covariance matrix C shape:", C.shape, f" = ({numNeurons} x {numNeurons})")
print()
print("  * Computed manually: C = (X_c @ X_c.T) / (T - 1)")
print("  * C[i,j] = covariance between neuron i and neuron j across time/dirs.")
print("  * Diagonal = each neuron's variance.")
print("  * Off-diagonal structure:")
print("    - Positive blocks: neurons that co-activate for the same directions.")
print("    - Negative values: neurons with opposite direction preferences.")
print("    - Block structure (if neurons sorted by preferred direction)")
print("      would show ~8 clusters aligned with the 8 reach directions.")

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Prob 1  --  1.2.1 Mean-centred data  &  1.2.2 Covariance matrix',
             fontsize=11, color='white')

# 1.2.1 heatmap
ax = axes[0]
im = ax.imshow(X_c, aspect='auto', cmap='RdBu_r', interpolation='nearest')
plt.colorbar(im, ax=ax, pad=0.02)
ax.set_title('1.2.1  Mean-centred X_c', fontsize=10)
ax.set_xlabel('Time bins (all 8 directions concatenated)', fontsize=9)
ax.set_ylabel('Neurons', fontsize=9)
for d in range(1, numDirs):
    ax.axvline(d * numTimebins - 0.5, color='yellow', linewidth=0.6, alpha=0.5)
ax.grid(False)

# 1.2.2 covariance heatmap
ax = axes[1]
im2 = ax.imshow(C, aspect='auto', cmap='RdBu_r')
plt.colorbar(im2, ax=ax, pad=0.02)
ax.set_title('1.2.2  Covariance matrix C', fontsize=10)
ax.set_xlabel('Neurons', fontsize=9)
ax.set_ylabel('Neurons', fontsize=9)
ax.grid(False)

plt.tight_layout()
plt.show()
