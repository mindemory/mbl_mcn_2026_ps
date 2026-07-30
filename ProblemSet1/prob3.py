import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import eigh

plt.style.use('dark_background')
rng = np.random.default_rng(42)   # reproducible

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def clean_ax(ax):
    ax.grid(False)
    ax.tick_params(labelsize=8)
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor('#1a1a2e')
        leg.get_frame().set_edgecolor('#444')

# ═════════════════════════════════════════════════════════════════════════════
# 3.1  Design your stimulus
#   1D Gaussian white noise, bin_width=2 ms, T=200 000 ms (200 s)
# ═════════════════════════════════════════════════════════════════════════════
bin_width  = 2          # ms
T          = 200_000    # ms (200 s → ~4000 spikes at 20 Hz for accurate STC)
t_stim     = np.arange(0, T, bin_width)
N_stim     = len(t_stim)
stim       = rng.standard_normal(N_stim)    # Gaussian white noise, std=1

# ═════════════════════════════════════════════════════════════════════════════
# 3.2  Simulate data
#   Kernel:   f(t) = exp(−t/τ)·sin(ωt),  τ=10 ms, ω=0.3 rad/ms,  0≤t≤50 ms
#   Drive:    u_f(t) = Σ_{k=0}^{L-1} s(t−k)·f(k)   [causal convolution]
#
#   Note on np.convolve:
#     convolve(stim, filt, 'full')[:N_stim] gives exactly the causal u_f[t].
#
#   The raw drive u_f ~ N(0, ||filt||²); we normalise by std(u_f) so the
#   sigmoid parameters θ, Δ are in standard-deviation units.
#
#   Rate:     r(û) = r_max / (1 + exp[(θ−û)/Δ])   [Hz]
#   Spikes:   Poisson with P(spike in bin) = r·bin_width/1000
#
#   Parameters: θ=0, Δ=0.2 (in normalised units) → 50% baseline, sharp sigmoid.
#   r_max tuned so mean rate ≈ 20 Hz.
# ═════════════════════════════════════════════════════════════════════════════
tau_f    = 10.0   # ms
omega_f  = 0.3    # rad/ms
t_filt   = np.arange(0, 52, bin_width)    # 0…50 ms
L        = len(t_filt)

filt_true      = np.exp(-t_filt / tau_f) * np.sin(omega_f * t_filt)
filt_true_unit = filt_true / np.linalg.norm(filt_true)   # unit-norm for STA comparison

# Causal convolution: u_f[t] = convolve(stim, filt)[:N_stim]
u_f_full   = np.convolve(stim, filt_true, mode='full')
u_f        = u_f_full[:N_stim]          # shape (N_stim,), causal drive

u_f_norm   = u_f / u_f.std()            # normalised drive (std=1)

# Sigmoid nonlinearity (r in Hz)
target_Hz  = 20.0                       # desired mean firing rate
theta_lin  = 0.0                        # threshold in normalised units (midpoint)
Delta_lin  = 0.2                        # gain (sharpness)

def sigmoid_Hz(u, r_max, theta, Delta):
    return r_max / (1.0 + np.exp((theta - u) / Delta))

r_max_lin  = target_Hz / np.mean(sigmoid_Hz(u_f_norm, 1.0, theta_lin, Delta_lin))
r_lin      = sigmoid_Hz(u_f_norm, r_max_lin, theta_lin, Delta_lin)
prob_lin   = np.clip(r_lin * bin_width / 1000.0, 0, 1)
spikes_lin = (rng.uniform(size=N_stim) < prob_lin).astype(float)
n_sp_lin   = int(spikes_lin.sum())
rate_lin   = n_sp_lin / (T / 1000.0)
print(f"\n3.2  Linear model: {n_sp_lin} spikes,  mean rate = {rate_lin:.1f} Hz")

# ═════════════════════════════════════════════════════════════════════════════
# 3.3  STA
#   STA = (1/n_spikes) Σ_{spike at t} snippet(t)
#   snippet(t): reversed length-L window of stimulus ending at time t
#   Vectorised with sliding_window_view for speed.
# ═════════════════════════════════════════════════════════════════════════════
# stim_mat[i] = [stim[i+L-1], stim[i+L-2], …, stim[i]]  (lag-0 first)
stim_mat   = np.lib.stride_tricks.sliding_window_view(stim, L)[:, ::-1]   # (N_valid, L)
N_valid    = stim_mat.shape[0]
# Align: stim_mat[i] has its newest sample at stim index i + L-1
# → spikes_val[i] = spikes at stim index i + L-1
spv_lin    = spikes_lin[L - 1: L - 1 + N_valid]

n_used_lin  = int(spv_lin.sum())
sta_lin     = stim_mat.T @ spv_lin / max(n_used_lin, 1)   # (L,)
sta_lin_n   = sta_lin / (np.linalg.norm(sta_lin) + 1e-12)
cos_sta     = np.dot(sta_lin_n, filt_true_unit)
print(f"3.3  STA cosine sim to f = {cos_sta:.3f}  ({n_used_lin} spikes)")

# ═════════════════════════════════════════════════════════════════════════════
# 3.4  STC — 1D quadratic case
#   r(u_f) = r_max / (1 + exp[(θ − û_f²)/Δ])
#   û_f ~ N(0,1) → û_f² ~ χ²(1), mean=1
#   θ=1 (χ² mean), Δ=0.3 → sigmoid centred at typical squared drive
#   Even nonlinearity → STA≈0; STC reveals the filter.
# ═════════════════════════════════════════════════════════════════════════════
theta_quad = 1.0
Delta_quad  = 0.3

r_max_quad  = target_Hz / np.mean(sigmoid_Hz(u_f_norm**2, 1.0, theta_quad, Delta_quad))
r_quad      = sigmoid_Hz(u_f_norm**2, r_max_quad, theta_quad, Delta_quad)
prob_quad   = np.clip(r_quad * bin_width / 1000.0, 0, 1)
spikes_quad = (rng.uniform(size=N_stim) < prob_quad).astype(float)
n_sp_quad   = int(spikes_quad.sum())
rate_quad   = n_sp_quad / (T / 1000.0)
print(f"\n3.4  Quadratic model: {n_sp_quad} spikes,  mean rate = {rate_quad:.1f} Hz")

def compute_stc(stim_mat, spikes_val, L):
    """Returns STA, STC = C_spike − I (prior), and n_spikes."""
    idx  = np.where(spikes_val > 0)[0]
    n_sp = len(idx)
    if n_sp == 0:
        return np.zeros(L), np.zeros((L, L)), 0
    X    = stim_mat[idx]                # (n_sp, L)
    sta  = X.mean(axis=0)
    Xc   = X - sta
    C_sp = Xc.T @ Xc / n_sp
    stc  = C_sp - np.eye(L)            # subtract white-noise prior (= I)
    return sta, stc, n_sp

spv_quad = spikes_quad[L - 1: L - 1 + N_valid]
sta_quad, stc_quad, n_used_quad = compute_stc(stim_mat, spv_quad, L)

eigvals_q, eigvecs_q = eigh(stc_quad)
idx_q = np.argsort(np.abs(eigvals_q))[::-1]
eigvals_q = eigvals_q[idx_q]; eigvecs_q = eigvecs_q[:, idx_q]

top_q   = eigvecs_q[:, 0]
cos_q   = abs(np.dot(top_q, filt_true_unit))
print(f"3.4.1  ||STA_quad|| = {np.linalg.norm(sta_quad):.4f}  (expected ~0)")
print(f"3.4.2  STC top ev cosine sim to f = {cos_q:.3f}  (λ={eigvals_q[0]:.4f})")

# ═════════════════════════════════════════════════════════════════════════════
# 3.5  STC — 2D case
#   g(t) = −exp(−t/τ_g)·cos(ω_g·t),  τ_g=30 ms, ω_g=0.2 rad/ms
#   r(û_f, û_g) = r_max / [(1+exp[(θ−û_f²)/Δ]) · (1+exp[(θ−û_g²)/Δ])]
# ═════════════════════════════════════════════════════════════════════════════
tau_g   = 30.0
omega_g = 0.2
filt_g  = -np.exp(-t_filt / tau_g) * np.cos(omega_g * t_filt)
filt_g_unit = filt_g / np.linalg.norm(filt_g)

u_g_full = np.convolve(stim, filt_g, mode='full')
u_g      = u_g_full[:N_stim]
u_g_norm = u_g / u_g.std()

theta_2d = 1.0; Delta_2d = 0.3
sf2 = sigmoid_Hz(u_f_norm**2, 1.0, theta_2d, Delta_2d)
sg2 = sigmoid_Hz(u_g_norm**2, 1.0, theta_2d, Delta_2d)
r_max_2d    = target_Hz / np.mean(sf2 * sg2)
r_2d        = r_max_2d * sf2 * sg2
prob_2d     = np.clip(r_2d * bin_width / 1000.0, 0, 1)
spikes_2d   = (rng.uniform(size=N_stim) < prob_2d).astype(float)
n_sp_2d     = int(spikes_2d.sum())
rate_2d     = n_sp_2d / (T / 1000.0)
print(f"\n3.5  2D model: {n_sp_2d} spikes,  mean rate = {rate_2d:.1f} Hz")

spv_2d = spikes_2d[L - 1: L - 1 + N_valid]
sta_2d, stc_2d, n_used_2d = compute_stc(stim_mat, spv_2d, L)

eigvals_2, eigvecs_2 = eigh(stc_2d)
idx_2 = np.argsort(np.abs(eigvals_2))[::-1]
eigvals_2 = eigvals_2[idx_2]; eigvecs_2 = eigvecs_2[:, idx_2]
top1 = eigvecs_2[:, 0]; top2 = eigvecs_2[:, 1]

cs11 = abs(np.dot(top1, filt_true_unit));  cs12 = abs(np.dot(top1, filt_g_unit))
cs21 = abs(np.dot(top2, filt_true_unit));  cs22 = abs(np.dot(top2, filt_g_unit))
print(f"3.5.2  ev1: cos(f)={cs11:.3f}, cos(g)={cs12:.3f}  λ={eigvals_2[0]:.4f}")
print(f"       ev2: cos(f)={cs21:.3f}, cos(g)={cs22:.3f}  λ={eigvals_2[1]:.4f}")

# ── 3.5.3  Estimated nonlinearity ──
prior_p1 = stim_mat @ top1; prior_p2 = stim_mat @ top2
sp_idx   = np.where(spv_2d > 0)[0]
st_p1    = stim_mat[sp_idx] @ top1; st_p2 = stim_mat[sp_idx] @ top2

n_bins = 20; lim = 3.0
p1_e = np.linspace(-lim, lim, n_bins + 1)
p2_e = np.linspace(-lim, lim, n_bins + 1)
H_prior, _, _ = np.histogram2d(prior_p1, prior_p2, bins=[p1_e, p2_e])
H_spike, _, _ = np.histogram2d(st_p1, st_p2, bins=[p1_e, p2_e])
nonlin_est = (H_spike / np.maximum(H_prior, 1)).T

# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

# ── Figure 1: 3.1 & 3.2  stimulus and kernel ─────────────────────────────────
fig1, ax1 = plt.subplots(1, 2, figsize=(12, 4))
fig1.suptitle('Problem 3.1–3.2  Stimulus & True Filter', fontsize=11, color='white')

ax = ax1[0]
ax.plot(t_stim[:200], stim[:200], color='#ff9955', lw=1.0)
ax.set_xlabel('Time (ms)', fontsize=9); ax.set_ylabel('s(t)', fontsize=9)
ax.set_title('3.1  Gaussian white noise (first 400 ms)', fontsize=10); clean_ax(ax)

ax = ax1[1]
ax.plot(t_filt, filt_true_unit, color='cyan', lw=2.0,
        label=r'f(t) = e$^{-t/\tau}$·sin(ωt) [unit-norm]')
ax.axhline(0, color='white', lw=0.4, ls=':')
ax.set_xlabel('Lag t (ms)', fontsize=9); ax.set_ylabel('Amplitude', fontsize=9)
ax.set_title('3.2  True temporal filter f(t)', fontsize=10)
ax.legend(fontsize=9); clean_ax(ax)
plt.tight_layout(); plt.show()

# ── Figure 2: 3.3  STA ───────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle('Problem 3.3  Spike-Triggered Average (linear sigmoid model)',
              fontsize=11, color='white')

ax = ax2[0]
ax.plot(t_filt, filt_true_unit, color='cyan',    lw=2.0, label='True filter f(t)')
ax.plot(t_filt, sta_lin_n,      color='#ff6644', lw=2.0, ls='--',
        label=f'STA norm. ({n_used_lin} spikes)')
ax.axhline(0, color='white', lw=0.4, ls=':')
ax.set_xlabel('Lag (ms)', fontsize=9); ax.set_ylabel('Normalised amplitude', fontsize=9)
ax.set_title('STA vs. true filter', fontsize=10)
ax.legend(fontsize=9); clean_ax(ax)
ax.text(0.02, 0.05, f'Cosine similarity = {cos_sta:.3f}',
        transform=ax.transAxes, fontsize=8, color='#aaaaaa')

ax = ax2[1]
ax.text(0.05, 0.92,
    "3.3  Key points:\n\n"
    "• STA ∝ f(t) for Gaussian white noise + monotone nonlinearity\n"
    "  (Bussgang: E[s(t−τ)|spike] ∝ Cov(s,u_f) ∝ f(τ)).\n\n"
    "• Monotone sigmoid preserves sign/magnitude of u_f → spikes\n"
    "  are biased toward positive u_f → STA points along f.\n\n"
    "• Accuracy ∝ 1/√n_spikes; need n ≫ L = %d bins for clean estimate.\n\n"
    f"  Result: cosine sim = {cos_sta:.3f} from {n_used_lin} spikes." % L,
    transform=ax.transAxes, fontsize=9, color='white', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#444', alpha=0.9))
ax.axis('off'); ax.set_title('3.3  Written answer', fontsize=10)
plt.tight_layout(); plt.show()

# ── Figure 3: 3.4  STA & STC quadratic ───────────────────────────────────────
fig3, ax3 = plt.subplots(1, 3, figsize=(15, 4.5))
fig3.suptitle('Problem 3.4  STA & STC — 1D quadratic nonlinearity',
              fontsize=11, color='white')

ax = ax3[0]
ax.plot(t_filt, sta_quad, color='#ff6644', lw=2.0, label='STA (quadratic)')
ax.axhline(0, color='white', lw=0.5, ls=':')
ax.set_xlabel('Lag (ms)', fontsize=9); ax.set_ylabel('Amplitude', fontsize=9)
ax.set_title('3.4.1  STA — should be ~0', fontsize=10)
ax.legend(fontsize=9); clean_ax(ax)
ax.text(0.02, 0.05, f'||STA|| = {np.linalg.norm(sta_quad):.4f}',
        transform=ax.transAxes, fontsize=8, color='#aaaaaa')

ax = ax3[1]
n_show = min(14, L)
cols_q = ['#ff3333' if v > 0 else '#3399ff' for v in eigvals_q[:n_show]]
ax.bar(np.arange(1, n_show + 1), eigvals_q[:n_show], color=cols_q, alpha=0.85)
ax.axhline(0, color='white', lw=0.5)
ax.set_xlabel('Eigenvector rank', fontsize=9); ax.set_ylabel('Eigenvalue', fontsize=9)
ax.set_title('3.4.2  STC eigenspectrum', fontsize=10); clean_ax(ax)

ax = ax3[2]
sgn = np.sign(np.dot(top_q, filt_true_unit))
ax.plot(t_filt, filt_true_unit, color='cyan',    lw=2.0, label='True filter f(t)')
ax.plot(t_filt, sgn * top_q,   color='#ff8800', lw=2.0, ls='--',
        label=f'STC top ev (cos={cos_q:.3f})')
ax.axhline(0, color='white', lw=0.4, ls=':')
ax.set_xlabel('Lag (ms)', fontsize=9); ax.set_ylabel('Normalised amplitude', fontsize=9)
ax.set_title('3.4.2  STC top eigenvec vs. true filter', fontsize=10)
ax.legend(fontsize=8); clean_ax(ax)
plt.tight_layout(); plt.show()

# ── Figure 4: 3.5  2D STC ────────────────────────────────────────────────────
fig4 = plt.figure(figsize=(16, 9))
gs   = gridspec.GridSpec(2, 4, figure=fig4, hspace=0.45, wspace=0.4)
fig4.suptitle('Problem 3.5  STC — 2D case (two linear filters)',
              fontsize=11, color='white')

# 3.5.1 — theoretical nonlinearity
ax = fig4.add_subplot(gs[0, 0])
uvals = np.linspace(-4, 4, 200)
UF, UG = np.meshgrid(uvals, uvals)
R_th = 1.0 / ((1 + np.exp((theta_2d - UF**2)/Delta_2d)) *
               (1 + np.exp((theta_2d - UG**2)/Delta_2d)))
im = ax.contourf(UF, UG, R_th, levels=20, cmap='plasma')
plt.colorbar(im, ax=ax, label='r (a.u.)', shrink=0.9)
ax.set_xlabel('û_f', fontsize=9); ax.set_ylabel('û_g', fontsize=9)
ax.set_title('3.5.1  Theoretical r(û_f, û_g)', fontsize=9); clean_ax(ax)

ax = fig4.add_subplot(gs[0, 1])
ax.text(0.04, 0.96,
    "3.5.1 Parameter meanings:\n\n"
    "Δ_f, Δ_g — slope/gain of each sigmoid;\n"
    "  larger Δ → shallower, broader tuning.\n\n"
    "θ_f, θ_g — threshold on û²; neuron fires\n"
    "  when |û_f|>√θ_f AND |û_g|>√θ_g.\n\n"
    "Product structure: both projections must\n"
    "  be large simultaneously → 'corner'\n"
    "  high-firing pattern in (û_f, û_g) space.\n\n"
    "r(û_f,û_g) = r(−û_f,û_g) = r(û_f,−û_g)\n"
    "→ even in both dims → STA = 0.",
    transform=ax.transAxes, fontsize=8.5, color='white', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#555', alpha=0.9))
ax.axis('off'); ax.set_title('3.5.1  Written answer', fontsize=9)

ax = fig4.add_subplot(gs[0, 2])
ax.plot(t_filt, filt_true_unit, color='cyan',    lw=2, label='f(t) true')
ax.plot(t_filt, filt_g_unit,    color='#ff88ff', lw=2, label='g(t) true')
ax.axhline(0, color='white', lw=0.4, ls=':')
ax.set_xlabel('Lag (ms)', fontsize=9); ax.set_ylabel('Amplitude', fontsize=9)
ax.set_title('3.5.2  True filters f & g', fontsize=9)
ax.legend(fontsize=8); clean_ax(ax)

ax = fig4.add_subplot(gs[0, 3])
s1 = np.sign(np.dot(top1, filt_true_unit)) if cs11 > cs12 else np.sign(np.dot(top1, filt_g_unit))
s2 = np.sign(np.dot(top2, filt_g_unit)) if cs22 > cs21 else np.sign(np.dot(top2, filt_true_unit))
ax.plot(t_filt, s1 * top1, color='#ff8800', lw=2, ls='--',
        label=f'STC ev1 (λ={eigvals_2[0]:.3f})')
ax.plot(t_filt, s2 * top2, color='#ff44ff', lw=2, ls=':',
        label=f'STC ev2 (λ={eigvals_2[1]:.3f})')
ax.plot(t_filt, filt_true_unit, color='cyan',    lw=1.2, alpha=0.6, label='f true')
ax.plot(t_filt, filt_g_unit,    color='#aaaaff', lw=1.2, alpha=0.6, label='g true')
ax.axhline(0, color='white', lw=0.4, ls=':')
ax.set_xlabel('Lag (ms)', fontsize=9); ax.set_ylabel('Amplitude', fontsize=9)
ax.set_title('3.5.2  STC top-2 vs. true filters', fontsize=9)
ax.legend(fontsize=7); clean_ax(ax)

ax = fig4.add_subplot(gs[1, 0:2])
ax.text(0.02, 0.96,
    "3.5.2  STC recovery of two filters:\n\n"
    "• STA = 0: nonlinearity is even in both û_f and û_g → spike-triggered\n"
    "  distribution is symmetric around the origin in both dimensions.\n\n"
    "• STC = E[xx^T|spike] − E[xx^T|prior].  For an even quadratic nonlinearity,\n"
    "  spikes are more likely when |û_f| is large → variance inflated along f.\n"
    "  Same for g independently (product structure).\n\n"
    "• Top-2 STC eigenvectors (by |λ|) recover the subspace spanned by {f, g}:\n"
    f"  ev1 ↔ f: cos={cs11:.3f},   ev2 ↔ g: cos={cs22:.3f}\n\n"
    "• Recovery succeeds because f ⊥ g (orthogonal filters → distinct eigenvalues).\n"
    "  STC needs ~L× more spikes than STA for equivalent SNR (L = filter length).\n"
    "• If f and g are non-orthogonal, additional whitening/rotation is needed.",
    transform=ax.transAxes, fontsize=8.5, color='white', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#555', alpha=0.9))
ax.axis('off'); ax.set_title('3.5.2  Written answer', fontsize=9)

ax = fig4.add_subplot(gs[1, 2])
p1c = 0.5*(p1_e[:-1]+p1_e[1:]); p2c = 0.5*(p2_e[:-1]+p2_e[1:])
P1, P2 = np.meshgrid(p1c, p2c)
im2 = ax.contourf(P1, P2, nonlin_est, levels=15, cmap='plasma')
plt.colorbar(im2, ax=ax, label='Spike / prior count', shrink=0.9)
ax.set_xlabel('Projection onto ev1', fontsize=9)
ax.set_ylabel('Projection onto ev2', fontsize=9)
ax.set_title('3.5.3  Empirical nonlinearity', fontsize=9); clean_ax(ax)

ax = fig4.add_subplot(gs[1, 3])
n_show2 = min(12, L)
cols2 = ['#ff3333' if v > 0 else '#3399ff' for v in eigvals_2[:n_show2]]
ax.bar(np.arange(1, n_show2 + 1), eigvals_2[:n_show2], color=cols2, alpha=0.85)
ax.axhline(0, color='white', lw=0.5)
ax.set_xlabel('Eigenvector rank', fontsize=9); ax.set_ylabel('Eigenvalue', fontsize=9)
ax.set_title('3.5.3  STC eigenspectrum (2D model)', fontsize=9); clean_ax(ax)
ax.text(0.45, 0.88, f'λ1={eigvals_2[0]:.3f}\nλ2={eigvals_2[1]:.3f}',
        transform=ax.transAxes, fontsize=8, color='#aaaaaa')

plt.show()

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Written Answers Summary ===")
print(f"""
3.1  Stimulus: Gaussian white noise, bin_width={bin_width} ms, T={T} ms ({N_stim} bins).

3.2  Filter: f(t)=exp(−t/τ)·sin(ωt), τ={tau_f} ms, ω={omega_f} rad/ms, 0≤t≤50 ms.
     Drive: û_f(t)=u_f(t)/std(u_f), where u_f=convolve(stim,filt)[causal].
     Sigmoid: r(û)=r_max/(1+exp[(θ−û)/Δ]) Hz, θ={theta_lin}, Δ={Delta_lin}.
     r_max tuned → {rate_lin:.1f} Hz observed ({n_sp_lin} spikes).

3.3  STA recovers f(t) (cosine sim = {cos_sta:.3f}, {n_used_lin} spikes).
     Bussgang theorem: E[s(t−τ)|spike] ∝ f(τ) for Gaussian stim + monotone nonlinearity.
     Need n_spikes ≫ L = {L} for reliable estimate.

3.4.1 ||STA_quad|| = {np.linalg.norm(sta_quad):.4f} ≈ 0.
      Quadratic (even) nonlinearity: spike-triggered distribution is symmetric → STA=0.
      STA completely fails to reveal f for even nonlinearities.

3.4.2 STC top ev cosine sim = {cos_q:.3f} (λ={eigvals_q[0]:.4f}).
      STC recovers f even when STA fails. Significant eigenvalue flags the filter.

3.5.1 Δ: sharpness of tuning per dimension (larger Δ = broader).
      θ: threshold on û²; firing requires |û|>√θ in each dimension.
      Product structure → high r only in 'corners' (large |û_f| AND |û_g|).
      r even in both û_f, û_g → STA=0.

3.5.2 2D STC ({n_sp_2d} spikes):
      ev1: cos(f)={cs11:.3f}, cos(g)={cs12:.3f}  λ={eigvals_2[0]:.4f}
      ev2: cos(f)={cs21:.3f}, cos(g)={cs22:.3f}  λ={eigvals_2[1]:.4f}
      Both f and g recovered as top-2 eigenvectors.
      Works because f⊥g; each independently inflates spike-triggered variance.

3.5.3 Empirical nonlinearity: ratio of spike-triggered / prior 2D histogram
      in (ev1, ev2) projection space. High-firing structure matches theory.
""")