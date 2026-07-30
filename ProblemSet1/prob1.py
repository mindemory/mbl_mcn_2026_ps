from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import os

plt.style.use('dark_background')

data = loadmat(os.path.join(os.getcwd(), 'ProblemSet1', 'HatsopoulosReachTask.mat'))
print(data.keys())

numNeurons   = data['numNeurons'].item()
numTimebins  = data['numTimebins'].item()
numTrials    = data['numTrials'].item()
uniqueDirections = np.unique(data['direction'].T)
numDirs      = len(uniqueDirections)

# ─────────────────────────────────────────────
# 1.1  Build 2-D data matrix  (numNeurons x numTimebins*numDirs)
# ─────────────────────────────────────────────
trial_avg = np.empty((numNeurons, numTimebins, numDirs))
for idir in uniqueDirections:
    idx = np.where(data['direction'].T == idir)[1]
    trial_avg[:, :, idir - 1] = np.mean(data['firingRate'][:, :, idx], axis=2)

X = trial_avg.reshape(numNeurons, numTimebins * numDirs)

# ─────────────────────────────────────────────
# 1.2  PCA
# ─────────────────────────────────────────────

# 1.2.1  Row mean-centre
X_c = X - X.mean(axis=1, keepdims=True)

# 1.2.2  Covariance matrix (no np.cov)
T = X_c.shape[1]
C = (X_c @ X_c.T) / (T - 1)

# 1.2.3  Eigendecomposition
eig_vals, eig_vecs = np.linalg.eigh(C)
order    = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[order]
eig_vecs = eig_vecs[:, order]

var_exp = eig_vals / eig_vals.sum()
cum_var = np.cumsum(var_exp)
n90     = int(np.searchsorted(cum_var, 0.90)) + 1
print(f"\n1.2.3  PCs needed for >=90% variance: {n90}")

# 1.2.4  Project onto top-3 PCs
W3  = eig_vecs[:, :3]
Z   = W3.T @ X_c
Z3d = Z.reshape(3, numTimebins, numDirs)

# 1.2.5  No centering
C_raw     = (X @ X.T) / (X.shape[1] - 1)
ev_raw, _ = np.linalg.eigh(C_raw)
ev_raw    = np.sort(ev_raw)[::-1]

# 1.2.6  Column-centering
X_col     = X - X.mean(axis=0, keepdims=True)
C_col     = (X_col @ X_col.T) / (X_col.shape[1] - 1)
ev_col, _ = np.linalg.eigh(C_col)
ev_col    = np.sort(ev_col)[::-1]

# 1.2.7  Z-scoring rows
row_std   = X.std(axis=1, keepdims=True)
row_std[row_std == 0] = 1
X_z       = (X - X.mean(axis=1, keepdims=True)) / row_std
C_z       = (X_z @ X_z.T) / (X_z.shape[1] - 1)
ev_z, _   = np.linalg.eigh(C_z)
ev_z      = np.sort(ev_z)[::-1]

# ─────────────────────────────────────────────
# 1.3  SVD  (X_c = U S V^T, thin)
# ─────────────────────────────────────────────
U, s, Vt = np.linalg.svd(X_c, full_matrices=False)

eig_from_svd = s**2 / (T - 1)

# 1.3.2  Project onto top-3 left singular vectors
Z_svd    = U[:, :3].T @ X_c
Z_svd_3d = Z_svd.reshape(3, numTimebins, numDirs)

# ─────────────────────────────────────────────
# 1.4  Linear decoders
#   Data matrix: (numTrials, numNeurons) — mean FR per trial
# ─────────────────────────────────────────────
FR       = data['firingRate']          # (numNeurons, numTimebins, numTrials)
dirs_all = data['direction'].flatten() # (numTrials,)

# Mean across time -> (numNeurons, numTrials) -> transpose (numTrials, numNeurons)
X_dec = FR.mean(axis=1).T             # (numTrials, numNeurons)
scaler = StandardScaler()
X_dec_sc = scaler.fit_transform(X_dec)

clf = LogisticRegression(max_iter=500, C=1.0)
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1.4.1  Even (left vs right) split: dirs 1-4 = left, 5-8 = right
def make_binary_labels(dirs, group1_dirs):
    return np.where(np.isin(dirs, group1_dirs), 0, 1)

# Define splits to test
splits = {
    '1-4 vs 5-8\n(even)':        ([1,2,3,4], [5,6,7,8]),
    '2-5 vs 6-8,1\n(rotated)':   ([2,3,4,5], [6,7,8,1]),
    '3-6 vs 7-8,1-2\n(rotated)': ([3,4,5,6], [7,8,1,2]),
    '1,3,5,7 vs 2,4,6,8\n(XOR)': ([1,3,5,7], [2,4,6,8]),
}

accs = {}
for name, (g1, g2) in splits.items():
    y = make_binary_labels(dirs_all, g1)
    sc = cross_val_score(clf, X_dec_sc, y, cv=cv, scoring='accuracy')
    accs[name] = (sc.mean(), sc.std())
    print(f"1.4  {name.replace(chr(10),' ')}  acc={sc.mean():.3f} +/- {sc.std():.3f}")

# 1.4.2 / 1.4.3  project trial-average onto PC1-PC2 to show decision boundary
# Use trial-averaged data projected onto PCs for visualization
X_dec_c  = X_dec - X_dec.mean(axis=0, keepdims=True)   # col-center per neuron (trial baseline)
Z_dec    = (eig_vecs[:, :2].T @ X_dec_c.T).T            # (numTrials, 2)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
cmap   = plt.cm.get_cmap('hsv', numDirs + 1)
colors = [cmap(i) for i in range(numDirs)]

def clean_ax(ax):
    ax.grid(False)
    ax.tick_params(labelsize=7)
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor('#1a1a2e')
        leg.get_frame().set_edgecolor('#444')

def clean_ax3d(ax):
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.set_facecolor('black')
    ax.tick_params(labelsize=6)
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor('#1a1a2e')
        leg.get_frame().set_edgecolor('#444')

pc_labels  = [f'PC{i+1} ({var_exp[i]*100:.1f}%)' for i in range(3)]
svd_pct    = s[:3]**2 / (s**2).sum() * 100
svd_labels = [f'U{i+1} ({svd_pct[i]:.1f}%)' for i in range(3)]

# ─────────────────────────────────────────────
# Single figure  4 rows x 3 cols
#   Row 0: heatmap | covariance | eigenspectrum+cumvar
#   Row 1: PCA 3-D | SVD 3-D   | singular values vs eigenvalues
#   Row 2: no-center | col-center | z-score
#   Row 3: decoder accuracy | PC1-PC2 (even split) | PC1-PC2 (XOR split)
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11))
fig.suptitle('Problem Set 1 – PCA, SVD & Linear Decoders', fontsize=12, color='white')

# --- Row 0 ---
ax = fig.add_subplot(4, 3, 1)
im = ax.imshow(X_c, aspect='auto', cmap='RdBu_r', interpolation='nearest')
plt.colorbar(im, ax=ax, pad=0.02)
ax.set_title('1.2.1  Mean-centred data', fontsize=9)
ax.set_xlabel('Time bins (all dirs)', fontsize=8)
ax.set_ylabel('Neurons', fontsize=8)
clean_ax(ax)

ax = fig.add_subplot(4, 3, 2)
im2 = ax.imshow(C, aspect='auto', cmap='RdBu_r')
plt.colorbar(im2, ax=ax, pad=0.02)
ax.set_title('1.2.2  Covariance matrix', fontsize=9)
ax.set_xlabel('Neurons', fontsize=8)
ax.set_ylabel('Neurons', fontsize=8)
clean_ax(ax)

ax = fig.add_subplot(4, 3, 3)
ax.scatter(range(1, len(eig_vals)+1), var_exp*100, s=12, color='cyan', label='Var exp (%)')
ax.axvline(n90, color='yellow', linestyle='--', linewidth=1, label=f'90% @ PC{n90}')
ax2 = ax.twinx()
ax2.plot(range(1, len(cum_var)+1), cum_var*100, color='magenta', linewidth=1.2, label='Cumul. var')
ax2.axhline(90, color='yellow', linestyle=':', linewidth=0.8)
ax2.set_ylabel('Cumulative var (%)', fontsize=7, color='magenta')
ax2.tick_params(labelsize=7, colors='magenta')
ax2.grid(False)
ax.set_title('1.2.3  Eigenspectrum & cumul. var', fontsize=9)
ax.set_xlabel('PC index', fontsize=8)
ax.set_ylabel('Var explained (%)', fontsize=8)
l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1+l2, lb1+lb2, fontsize=7)
clean_ax(ax)

# --- Row 1 ---
ax = fig.add_subplot(4, 3, 4, projection='3d')
for d in range(numDirs):
    ax.plot(Z3d[0,:,d], Z3d[1,:,d], Z3d[2,:,d],
            color=colors[d], linewidth=1.0, label=f'{uniqueDirections[d]:.0f}deg')
    ax.scatter(Z3d[0,0,d], Z3d[1,0,d], Z3d[2,0,d], color=colors[d], s=20)
ax.set_title('1.2.4  PCA 3-D trajectories', fontsize=9)
ax.set_xlabel(pc_labels[0], fontsize=7)
ax.set_ylabel(pc_labels[1], fontsize=7)
ax.set_zlabel(pc_labels[2], fontsize=7)
ax.legend(fontsize=5, ncol=2, loc='upper left')
clean_ax3d(ax)

ax = fig.add_subplot(4, 3, 5, projection='3d')
for d in range(numDirs):
    ax.plot(Z_svd_3d[0,:,d], Z_svd_3d[1,:,d], Z_svd_3d[2,:,d],
            color=colors[d], linewidth=1.0, label=f'{uniqueDirections[d]:.0f}deg')
    ax.scatter(Z_svd_3d[0,0,d], Z_svd_3d[1,0,d], Z_svd_3d[2,0,d], color=colors[d], s=20)
ax.set_title('1.3.2  SVD 3-D trajectories', fontsize=9)
ax.set_xlabel(svd_labels[0], fontsize=7)
ax.set_ylabel(svd_labels[1], fontsize=7)
ax.set_zlabel(svd_labels[2], fontsize=7)
ax.legend(fontsize=5, ncol=2, loc='upper left')
clean_ax3d(ax)

ax = fig.add_subplot(4, 3, 6)
ax.scatter(range(1, len(s)+1), s, s=12, color='orange', label='Singular values (s)')
ax2b = ax.twinx()
ax2b.scatter(range(1, len(eig_vals)+1), eig_vals, s=12, color='cyan', alpha=0.6, marker='^', label='Eigenvalues')
ax2b.scatter(range(1, len(eig_from_svd)+1), eig_from_svd, s=12, color='lime', alpha=0.6, marker='x', label='s^2/(T-1)')
ax2b.set_ylabel('Eigenvalue', fontsize=7, color='cyan')
ax2b.tick_params(labelsize=7, colors='cyan')
ax2b.grid(False)
ax.set_title('1.3.1  Singular values & eigenvalues', fontsize=9)
ax.set_xlabel('Component index', fontsize=8)
ax.set_ylabel('Singular value', fontsize=8, color='orange')
ax.tick_params(colors='orange', labelsize=7)
la, lba = ax.get_legend_handles_labels()
lb, lbb = ax2b.get_legend_handles_labels()
ax.legend(la+lb, lba+lbb, fontsize=7)
clean_ax(ax)

# --- Row 2: centering ---
ax = fig.add_subplot(4, 3, 7)
ax.scatter(range(1, len(ev_raw)+1), ev_raw,     s=10, color='red',    label='No centering')
ax.scatter(range(1, len(eig_vals)+1), eig_vals,  s=10, color='cyan',   alpha=0.6, label='Row-centered')
ax.set_title('1.2.5  No centering vs row-centered', fontsize=9)
ax.set_xlabel('PC index', fontsize=8); ax.set_ylabel('Eigenvalue', fontsize=8)
ax.legend(fontsize=7); clean_ax(ax)

ax = fig.add_subplot(4, 3, 8)
ax.scatter(range(1, len(ev_col)+1), ev_col,      s=10, color='orange', label='Col-centered')
ax.scatter(range(1, len(eig_vals)+1), eig_vals,  s=10, color='cyan',   alpha=0.6, label='Row-centered')
ax.set_title('1.2.6  Column-centering', fontsize=9)
ax.set_xlabel('PC index', fontsize=8); ax.set_ylabel('Eigenvalue', fontsize=8)
ax.legend(fontsize=7); clean_ax(ax)

ax = fig.add_subplot(4, 3, 9)
ax.scatter(range(1, len(ev_z)+1), ev_z,          s=10, color='lime',   label='Z-scored')
ax.scatter(range(1, len(eig_vals)+1), eig_vals,  s=10, color='cyan',   alpha=0.6, label='Row-centered')
ax.set_title('1.2.7  Z-scored rows', fontsize=9)
ax.set_xlabel('PC index', fontsize=8); ax.set_ylabel('Eigenvalue', fontsize=8)
ax.legend(fontsize=7); clean_ax(ax)

# --- Row 3: decoders (1.4) ---
ax = fig.add_subplot(4, 3, 10)
names = list(accs.keys())
means = [accs[n][0] for n in names]
stds  = [accs[n][1] for n in names]
bar_colors = ['#5599ff', '#5599ff', '#5599ff', '#ff5555']
bars = ax.bar(range(len(names)), means, yerr=stds, color=bar_colors,
              capsize=4, error_kw={'linewidth': 1.2, 'ecolor': 'white'})
ax.axhline(0.5, color='yellow', linestyle='--', linewidth=1, label='Chance (0.5)')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=6)
ax.set_ylabel('CV accuracy (5-fold)', fontsize=8)
ax.set_title('1.4.1/1.4.2/1.4.3  Decoder accuracy', fontsize=9)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=7)
clean_ax(ax)

# PC1-PC2 scatter coloured by even split (1.4.1)
ax = fig.add_subplot(4, 3, 11)
y_even = make_binary_labels(dirs_all, [1,2,3,4])
split_colors_even = np.where(y_even == 0, '#4488ff', '#ff4444')
ax.scatter(Z_dec[:, 0], Z_dec[:, 1], c=split_colors_even, s=12, alpha=0.7)
# Add direction centroids
for idir in uniqueDirections:
    mask = dirs_all == idir
    ax.scatter(Z_dec[mask,0].mean(), Z_dec[mask,1].mean(),
               marker='*', s=60, color=colors[idir-1], edgecolors='white', linewidth=0.5)
ax.set_title('1.4.1  Trials in PC space (even split)', fontsize=9)
ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=8)
ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', fontsize=8)
from matplotlib.patches import Patch
legend_els = [Patch(facecolor='#5599ff', label='Group 1 (dirs 1-4)'),
              Patch(facecolor='#ff5555', label='Group 2 (dirs 5-8)')]
leg = ax.legend(handles=legend_els, fontsize=7)
leg.get_frame().set_facecolor('#1a1a2e')
leg.get_frame().set_edgecolor('#444')
clean_ax(ax)

# PC1-PC2 scatter coloured by XOR split (1.4.3)
ax = fig.add_subplot(4, 3, 12)
y_xor = make_binary_labels(dirs_all, [1,3,5,7])
split_colors_xor = np.where(y_xor == 0, '#4488ff', '#ff4444')
ax.scatter(Z_dec[:, 0], Z_dec[:, 1], c=split_colors_xor, s=12, alpha=0.7)
for idir in uniqueDirections:
    mask = dirs_all == idir
    ax.scatter(Z_dec[mask,0].mean(), Z_dec[mask,1].mean(),
               marker='*', s=60, color=colors[idir-1], edgecolors='white', linewidth=0.5)
ax.set_title('1.4.3  Trials in PC space (XOR split)', fontsize=9)
ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=8)
ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', fontsize=8)
legend_els_xor = [Patch(facecolor='#5599ff', label='Group 1 (dirs 1,3,5,7)'),
                  Patch(facecolor='#ff5555', label='Group 2 (dirs 2,4,6,8)')]
leg = ax.legend(handles=legend_els_xor, fontsize=7)
leg.get_frame().set_facecolor('#1a1a2e')
leg.get_frame().set_edgecolor('#444')
clean_ax(ax)

plt.tight_layout(pad=0.8, h_pad=1.0, w_pad=0.8)
plt.show()

print("\n=== Written answers ===")
print(f"\n1.1    -> 2D matrix shape: {X.shape} = (numNeurons={numNeurons}, numTimebins*numDirs={numTimebins*numDirs})")
print("          Goal: average across trials per direction, then concatenate directions side-by-side.")
print("\n1.2.1  -> Row mean-centering removes each neuron's mean firing rate (baseline),")
print("          leaving only direction/time-modulated variance.")
print(f"\n1.2.3  -> {n90} PCs explain >=90% of variance (out of {numNeurons} neurons).")
print("          Data is intrinsically low-dimensional; most variance in a small subspace.")
print(f"\n1.2.4  -> 8 trajectories visible in PC1-PC2-PC3 space, forming a roughly ring-like")
print("          structure — consistent with circular reach direction geometry.")
print("\n1.2.5  -> Without row-centering: PC1 captures mean firing rate, not direction variance;")
print("          first eigenvalue dominates; remaining structure is hidden.")
print("\n1.2.6  -> Column-centering removes population-mean at each time bin.")
print("          Eigenvalues shift because the removed component is per-time, not per-neuron.")
print("          Rank is still <=min(N,T), but the spectrum shape differs.")
print("\n1.2.7  -> Z-scoring (row) normalises each neuron to unit variance.")
print("          Good when neurons differ in firing rate scale; bad when that scale matters,")
print("          or when silent neurons exist (their noise gets amplified).")
print("\n1.2.8  -> Row-centering good when neurons have a fixed baseline to remove.")
print("          Col-centering good when population-level drift dominates across time.")
print("          Z-scoring good when neuron scales are heterogeneous but biologically arbitrary.")
print(f"\n1.3.1  -> U: ({numNeurons}x{len(s)}) left singular vectors = PC directions (neuron space).")
print(f"          S: diagonal of singular values; eigenvalue_i = s_i^2 / (T-1).")
print(f"          V^T: ({len(s)}x{T}) right singular vectors = time/condition scores.")
print("          Columns of U are eigenvectors of C; s_i^2/(T-1) are eigenvalues of C.")
print("\n1.3.2  -> SVD and PCA trajectories are identical (up to sign flip per component).")
print("          Equivalence requires X to be row-mean-centred (so C = X_c X_c^T / (T-1)).")
for name, (mn, sd) in accs.items():
    print(f"\n1.4    [{name.replace(chr(10),' ')}]  CV acc = {mn:.3f} +/- {sd:.3f}")
print("\n1.4.1  -> Contiguous (even) splits are well-decoded: directions on one side vs other")
print("          form linearly separable clusters in neural space.")
print("\n1.4.2  -> Rotated contiguous splits also decode well — any half-plane split works.")
print("          This is consistent with a ring-like geometry in the PC subspace.")
print("\n1.4.3  -> XOR-like (alternating) split is NOT linearly separable in general,")
print("          because the two groups interleave around the ring.")
print("          A linear decoder cannot separate them if the representation is circular.")
print("          A nonlinear decoder could; its success would imply higher-dim structure.")
print("          This shows PCA/linear analysis captures ring geometry but not XOR structure.")
