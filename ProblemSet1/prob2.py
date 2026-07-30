import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.style.use('dark_background')

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def clean_ax(ax):
    ax.grid(False)
    ax.tick_params(labelsize=8)
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor('#1a1a2e')
        leg.get_frame().set_edgecolor('#444')

# ═══════════════════════════════════════════════
# 2.1  Leaky Integrate-and-Fire — F-I curve
#   C dV/dt = -gL(V-EL) + I
#   Threshold VT, reset VR  (VR < VT)
#
#   Analytical ISI from reset VR to threshold VT:
#     V(t) = EL + I/gL + (VR - EL - I/gL) * exp(-t/tau),  tau=C/gL
#     t_spike = tau * ln[(I/gL-(VR-EL)) / (I/gL-(VT-EL))]
#   Firing rate:  f [Hz] = 1000 / t_spike
#   Rheobase:     I_th = gL * (VT - EL)
# ═══════════════════════════════════════════════
def lif_rate(I_arr, C=1.0, gL=0.1, EL=-65.0, VT=-50.0, VR=-65.0):
    I_arr = np.asarray(I_arr, dtype=float)
    tau   = C / gL          # ms
    I_th  = gL * (VT - EL)
    r     = np.zeros_like(I_arr)
    m     = I_arr > I_th
    if m.any():
        B    = I_arr[m] / gL - (VR - EL)   # > 0
        A    = I_arr[m] / gL - (VT - EL)   # > 0, B > A
        r[m] = 1000.0 / (tau * np.log(B / A))
    return r

C0, gL0, EL0, VT0, VR0 = 1.0, 0.1, -65.0, -50.0, -65.0
I_arr = np.linspace(0, 6, 500)

# ═══════════════════════════════════════════════
# 2.2  Synaptic conductance LIF — F-gS curve
#   C dV/dt = -gL(V-EL) - gS(V-ES)
#           = -(gL+gS)(V - E_eff),  E_eff = (gL*EL + gS*ES)/(gL+gS)
#   Equivalent to LIF with g_eff = gL+gS, effective reversal E_eff
#   t_spike = tau_eff * ln[(E_eff-VR) / (E_eff-VT)]
#   Threshold: E_eff > VT  →  gS_th = gL*(VT-EL)/(ES-VT)
# ═══════════════════════════════════════════════
def lif_syn_rate(gS_arr, C=1.0, gL=0.1, EL=-65.0, ES=0.0, VT=-50.0, VR=-65.0):
    r = np.zeros(len(gS_arr))
    for i, gS in enumerate(gS_arr):
        g_eff = gL + gS
        E_eff = (gL*EL + gS*ES) / g_eff
        if E_eff <= VT:
            continue
        tau_eff = C / g_eff
        r[i]    = 1000.0 / (tau_eff * np.log((E_eff-VR) / (E_eff-VT)))
    return r

def simulate_syn_lif(gS, C=1.0, gL=0.1, EL=-65.0, ES=0.0, VT=-50.0, VR=-65.0,
                      dt=0.05, T=300.0):
    t = np.arange(0, T, dt)
    V = np.empty(len(t)); V[0] = EL
    for i in range(1, len(t)):
        V[i] = V[i-1] + dt/C * (-gL*(V[i-1]-EL) - gS*(V[i-1]-ES))
        if V[i] >= VT:
            V[i] = VR
    return t, V

gS_arr = np.linspace(0, 1.5, 400)
ES0    = 0.0          # AMPA reversal (mV), > VT so excitatory
gS_th  = gL0*(VT0-EL0)/(ES0-VT0)   # = 0.03 mS/cm²

# ═══════════════════════════════════════════════
# 2.3  QIF with adaptation — bifurcation analysis
#   dV/dt = V² + I - W
#   dW/dt = a(bV - W)
#
#   Nullclines:
#     V-null: W = V²+I  (parabola, shifts up with I)
#     W-null: W = bV    (line through origin, slope b)
#   Fixed points:  V²-bV+I = 0  →  V* = (b ± √(b²-4I))/2
#   Saddle-node:   I_SN = b²/4           (discriminant = 0)
#   Jacobian at lower FP V*₋:
#     J = [[2V*₋, -1], [ab, -a]]
#     Tr = 2V*₋ - a,   Det = a√(b²-4I) > 0
#   Stability:
#     Stable  when Tr < 0:  2V*₋ < a
#     Hopf (requires b≥a):  Tr=0 →  I_Hopf = a(2b-a)/4
#   Upper FP V*₊:  Det = -a√(b²-4I) < 0  →  always a saddle
#   Bogdanov-Takens (Hopf meets SN):  b=a,  I_BT = a²/4
# ═══════════════════════════════════════════════
a0 = 0.5; b0 = 1.0
I_SN0   = b0**2 / 4                 # 0.25
I_Hopf0 = a0*(2*b0 - a0) / 4       # 0.1875

print(f"\n2.3  a={a0}, b={b0}")
print(f"  Saddle-node:  I_SN   = {I_SN0:.4f}")
print(f"  Hopf (b>=a):  I_Hopf = {I_Hopf0:.4f}")
print(f"  BT point:     b=a={a0},  I_BT = {a0**2/4:.4f}")

def qif_fps(I, b):
    disc = b**2 - 4*I
    if disc < 0: return []
    if disc == 0: return [(b/2, b*b/2)]
    Vm = (b - np.sqrt(disc))/2; Vp = (b + np.sqrt(disc))/2
    return [(Vm, b*Vm), (Vp, b*Vp)]

# ─────────────────────────────────────────────
# Figure:  3 rows × 3 cols
#   Row 0: 2.1 F-I default | vary VT | vary VR
#   Row 1: 2.2 F-gS curve  | voltage traces | vary ES
#   Row 2: 2.3 phase plane | bifurc curves   | param space
# ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
fig.suptitle('Problem 2 – Integrate-and-Fire Neuron Models', fontsize=12, color='white')
axes = axes.flatten()

# ── 2.1  default F-I ──
ax = axes[0]
ax.plot(I_arr, lif_rate(I_arr), color='cyan', lw=2, label=f'VT={VT0}, VR={VR0}')
ax.axvline(gL0*(VT0-EL0), color='yellow', ls='--', lw=1,
           label=f'Rheobase = {gL0*(VT0-EL0):.1f} μA/cm²')
ax.set_ylim(0, 400)
ax.set_xlabel('I (μA/cm²)', fontsize=9); ax.set_ylabel('Rate (Hz)', fontsize=9)
ax.set_title('2.1  LIF F-I curve', fontsize=10)
ax.legend(fontsize=8); clean_ax(ax)

# ── 2.1  vary VT ──
ax = axes[1]
for VT_v, col in zip([-55., -50., -45.], ['#88ccff', '#5599ff', '#2255dd']):
    ax.plot(I_arr, lif_rate(I_arr, VT=VT_v), color=col, lw=1.8, label=f'VT={VT_v} mV')
ax.set_ylim(0, 400)
ax.set_xlabel('I (μA/cm²)', fontsize=9); ax.set_ylabel('Rate (Hz)', fontsize=9)
ax.set_title('2.1  Effect of threshold VT', fontsize=10)
ax.legend(fontsize=8); clean_ax(ax)

# ── 2.1  vary VR ──
ax = axes[2]
for VR_v, col in zip([-72., -65., -58.], ['#ffaa44', '#ff8822', '#ff5500']):
    ax.plot(I_arr, lif_rate(I_arr, VR=VR_v), color=col, lw=1.8, label=f'VR={VR_v} mV')
ax.set_ylim(0, 400)
ax.set_xlabel('I (μA/cm²)', fontsize=9); ax.set_ylabel('Rate (Hz)', fontsize=9)
ax.set_title('2.1  Effect of reset VR', fontsize=10)
ax.legend(fontsize=8); clean_ax(ax)

# ── 2.2  F-gS curve ──
ax = axes[3]
ax.plot(gS_arr, lif_syn_rate(gS_arr), color='lime', lw=2, label='ES=0 mV')
ax.axvline(gS_th, color='yellow', ls='--', lw=1,
           label=f'gS_th = {gS_th:.3f} mS/cm²')
ax.set_ylim(0, 500)
ax.set_xlabel('gS (mS/cm²)', fontsize=9); ax.set_ylabel('Rate (Hz)', fontsize=9)
ax.set_title('2.2  Synaptic LIF: F vs gS', fontsize=10)
ax.legend(fontsize=8); clean_ax(ax)

# ── 2.2  voltage traces ──
ax = axes[4]
gS_traces = [(0.035, '#44ff88', 'gS=0.035'), (0.08, '#00cc44', 'gS=0.080'), (0.3, '#008822', 'gS=0.300')]
for gS_v, col, lbl in gS_traces:
    t_tr, V_tr = simulate_syn_lif(gS_v)
    ax.plot(t_tr, V_tr, color=col, lw=0.9, alpha=0.9, label=lbl)
ax.axhline(VT0, color='yellow', ls=':', lw=0.8, label=f'VT={VT0} mV')
ax.set_ylim(-68, -42); ax.set_xlim(0, 300)
ax.set_xlabel('Time (ms)', fontsize=9); ax.set_ylabel('V (mV)', fontsize=9)
ax.set_title('2.2  Voltage traces (diff. gS)', fontsize=10)
ax.legend(fontsize=7); clean_ax(ax)

# ── 2.2  vary ES ──
ax = axes[5]
for ES_v, col in zip([0., 30., 80.], ['lime', '#88ffaa', '#ccffdd']):
    ax.plot(gS_arr, lif_syn_rate(gS_arr, ES=ES_v), color=col, lw=1.8, label=f'ES={ES_v} mV')
ax.set_ylim(0, 500)
ax.set_xlabel('gS (mS/cm²)', fontsize=9); ax.set_ylabel('Rate (Hz)', fontsize=9)
ax.set_title('2.2  Effect of reversal ES', fontsize=10)
ax.legend(fontsize=8); clean_ax(ax)

# ── 2.3  phase plane ──
ax = axes[6]
V_range = np.linspace(-1.5, 2.8, 500)
phase_cases = [
    (0.10, '#5599ff', 'I=0.10 (osc.)'),
    (0.20, '#44dd88', 'I=0.20 (silent)'),
    (I_SN0, '#ffcc00', f'I={I_SN0:.2f} (SN)'),
    (0.32,  '#ff4444', 'I=0.32 (spiking)'),
]
for I_p, col, lbl in phase_cases:
    ax.plot(V_range, V_range**2 + I_p, color=col, lw=1.5, label=lbl)
ax.plot(V_range, b0*V_range, color='white', lw=1.2, ls='--', alpha=0.65, label='W-null: W=bV')
# Mark fixed points
for I_p, col, _ in phase_cases[:3]:
    for j, (Vf, Wf) in enumerate(qif_fps(I_p, b0)):
        if j == 0:   # lower FP: filled circle
            ax.scatter([Vf], [Wf], color=col, marker='o', s=55, zorder=6,
                       edgecolors='white', linewidths=0.6)
        else:        # upper FP (saddle): x marker — no edgecolor on unfilled markers
            ax.scatter([Vf], [Wf], color=col, marker='x', s=60, zorder=6, linewidths=1.5)
ax.set_xlim(-1.5, 2.8); ax.set_ylim(-1.0, 5.0)
ax.set_xlabel('V', fontsize=9); ax.set_ylabel('W', fontsize=9)
ax.set_title(f'2.3  Phase plane (a={a0}, b={b0})', fontsize=10)
ax.legend(fontsize=6); clean_ax(ax)
# annotate circle/cross meaning
ax.text(0.02, 0.05, '●=stable/unstable FP   ✕=saddle',
        transform=ax.transAxes, fontsize=6, color='#aaaaaa')

# ── 2.3  bifurcation curves ──
ax = axes[7]
b_range = np.linspace(0, 2, 500)
I_SN_c   = b_range**2 / 4
I_Hopf_c = a0*(2*b_range - a0) / 4
ax.plot(b_range, I_SN_c,   color='#ff8800', lw=2.0, label='Saddle-node: I=b²/4')
ax.plot(b_range, I_Hopf_c, color='cyan',    lw=2.0, label='Hopf: I=a(2b−a)/4')
ax.scatter([a0], [a0**2/4], color='yellow', s=80, zorder=6,
           label=f'BT point (b=a={a0})')
ax.axhline(0, color='white', lw=0.4, ls=':')
# Shade regions
ax.fill_between(b_range, I_SN_c, 1.1,  alpha=0.12, color='#ff4444')
ax.fill_between(b_range, np.clip(I_Hopf_c, 0, None), I_SN_c,
                alpha=0.12, color='#4488ff')
ax.fill_between(b_range, 0, np.clip(I_Hopf_c, 0, None),
                where=(b_range >= a0), alpha=0.12, color='#44cc55')
ax.set_xlim(0, 2); ax.set_ylim(-0.05, 1.1)
ax.set_xlabel('b', fontsize=9); ax.set_ylabel('I', fontsize=9)
ax.set_title(f'2.3  Bifurcation curves (a={a0})', fontsize=10)
ax.legend(fontsize=7); clean_ax(ax)

# ── 2.3  parameter space map ──
ax = axes[8]
b_g = np.linspace(0, 2, 300); I_g = np.linspace(-0.05, 1.1, 300)
BB, II = np.meshgrid(b_g, I_g)
I_SN_g   = BB**2 / 4
I_Hopf_g = a0*(2*BB - a0) / 4
reg = np.zeros_like(II)                                              # 0 = tonic spiking
reg[(II <= I_SN_g) & (II > np.clip(I_Hopf_g, 0, None))] = 1        # 1 = silent
reg[(II <= I_SN_g) & (II <= I_Hopf_g) & (BB >= a0)]     = 2        # 2 = oscillatory
ax.contourf(BB, II, reg, levels=[-0.5, 0.5, 1.5, 2.5],
            colors=['#ff3333', '#4466ff', '#33cc55'], alpha=0.35)
ax.plot(b_range, I_SN_c,   color='#ff8800', lw=2)
ax.plot(b_range, I_Hopf_c, color='cyan',    lw=2)
ax.scatter([a0], [a0**2/4], color='yellow', s=80, zorder=6, label=f'BT (b=a={a0})')
ax.axhline(0, color='white', lw=0.4, ls=':')
ax.set_xlim(0, 2); ax.set_ylim(-0.05, 1.1)
ax.set_xlabel('b', fontsize=9); ax.set_ylabel('I', fontsize=9)
ax.set_title(f'2.3  (I, b) parameter space (a={a0})', fontsize=10)
patches = [mpatches.Patch(color='#ff3333', alpha=0.65, label='Tonic spiking (no FP)'),
           mpatches.Patch(color='#4466ff', alpha=0.65, label='Silent (stable FP)'),
           mpatches.Patch(color='#33cc55', alpha=0.65, label='Oscillatory (limit cycle)')]
leg = ax.legend(handles=patches, fontsize=7)
leg.get_frame().set_facecolor('#1a1a2e'); leg.get_frame().set_edgecolor('#444')
clean_ax(ax)

plt.tight_layout(pad=0.8, h_pad=1.0, w_pad=0.8)
plt.show()

# ─────────────────────────────────────────────
# Written answers (in question order)
# ─────────────────────────────────────────────
print("\n=== Written answers ===")

print(f"\n2.1 -> LIF analytical F-I curve:")
print(f"       Rheobase: I_th = gL*(VT-EL) = {gL0*(VT0-EL0):.2f} μA/cm²")
print(f"       For I > I_th: f = 1000 / [tau*ln(B/A)]  [Hz]")
print(f"         B = I/gL-(VR-EL),  A = I/gL-(VT-EL),  tau=C/gL={C0/gL0:.0f} ms")
print(f"       Effect of VT: higher VT → higher I_th → curve shifts right; same rate shape")
print(f"       Effect of VR: lower VR → farther from threshold after reset → longer ISI → lower rate")
print(f"       Both VT and VR set the rheobase; VR additionally controls slope at high I.")

print(f"\n2.2 -> Synaptic LIF (ES={ES0} mV, EL={EL0} mV, gL={gL0} mS/cm²):")
print(f"       Equivalent to LIF with g_eff=gL+gS, E_eff=(gL*EL+gS*ES)/(gL+gS)")
print(f"       Threshold: E_eff>VT → gS_th = gL*(VT-EL)/(ES-VT) = {gS_th:.4f} mS/cm²")
print(f"       f = 1000/[tau_eff*ln((E_eff-VR)/(E_eff-VT))],  tau_eff=C/(gL+gS)")
print(f"       As gS→∞: E_eff→ES, tau_eff→0 → rate diverges (but biologically saturates)")
print(f"       Higher ES: lower gS_th, steeper F-gS curve (more depolarising drive)")

print(f"\n2.3 -> QIF with adaptation (a={a0}, b={b0}):")
print(f"       Fixed points: V*=(b±√(b²-4I))/2")
print(f"         2 FPs when I < b²/4 = {I_SN0:.4f}")
print(f"         No FPs when I > b²/4  (→ tonic spiking: V→∞ repeatedly)")
print(f"         Saddle-node bifurcation at I_SN = b²/4 = {I_SN0:.4f}")
print(f"       Upper FP V*₊:  Det=−a√(b²-4I) < 0  →  always a saddle")
print(f"       Lower FP V*₋:  Det=+a√(b²-4I) > 0  →  stable or unstable focus/node")
print(f"         Trace = 2V*₋ − a;  stable when Trace < 0")
print(f"         Hopf bifurcation (requires b≥a): Trace=0 → I_Hopf = a(2b-a)/4 = {I_Hopf0:.4f}")
print(f"         For b < a: lower FP is always stable when it exists (no Hopf)")
print(f"       Bogdanov-Takens point: Hopf meets saddle-node at b=a={a0}, I_BT=a²/4={a0**2/4:.4f}")
print(f"       Regions in (I, b) space for a={a0}:")
print(f"         I > b²/4                →  tonic spiking (no fixed point)")
print(f"         a(2b-a)/4 < I < b²/4   →  silent (stable FP, Trace < 0)")
print(f"         0 < I < a(2b-a)/4, b≥a →  subthreshold oscillations (Hopf limit cycle)")

# ═══════════════════════════════════════════════
# 2.4  Fun with planar dynamics
#   System: dV/dt = f(V,n),  dn/dt = g(V,n)
#   Assumptions (unless extra credit): fn<0, gn<0, gv>0
#
# 2.4.1  Stability at an equilibrium with fV < 0
#   Jacobian J = [[fV, fn], [gV, gn]]
#   Tr(J) = fV + gn < 0   (both negative by assumption)
#   Det(J) = fV*gn - fn*gV
#          = (neg)(neg) - (neg)(pos) = pos + pos > 0
#   Both conditions for stability satisfied → equilibrium is stable. ✓
#
# 2.4.2  Middle-branch equilibrium has fV > 0
#   On the cubic V-nullcline f=0:
#     By implicit differentiation: df = fV*dV + fn*dn = 0
#     → slope of V-nullcline: dn/dV|_{f=0} = -fV/fn
#   The cubic has left/right branches (decreasing: dn/dV<0) and middle branch (increasing: dn/dV>0).
#   Middle branch: dn/dV > 0  → -fV/fn > 0
#   Since fn < 0:  -fV/fn > 0  ⟺  fV > 0. ✓
#
# 2.4.3  Middle-branch equilibrium is a saddle (when n-nullcline slope < V-nullcline slope)
#   V-nullcline slope: dn/dV|_{V-null} = -fV/fn  (positive on middle branch, fV>0, fn<0)
#   n-nullcline g=0 slope: dn/dV|_{g=0} = -gV/gn  (positive, since gV>0, gn<0)
#   Condition: slope(n-null) < slope(V-null):  -gV/gn < -fV/fn
#   Rearranging (gn<0, fn<0, both negative denominators flip inequality on multiply):
#     gV*fn > fV*gn   →   fV*gn - gV*fn < 0   →   Det(J) < 0
#   Det(J) < 0 → one positive, one negative eigenvalue → SADDLE. ✓
#   (If slope(n-null) > slope(V-null): Det>0, Tr=fV+gn could be >0 or <0 → unstable node/focus)
#
# 2.4.4  Extra Credit: No limit cycles when fn*gV > 0 for all V, n
#   Use Bendixson's criterion: div(F) = fV + gn
#   We need to show this doesn't change sign over a simply connected region.
#   Actually, with the Dulac function B = 1/(fn*gV):
#   When fn>0, gV>0 everywhere:
#     div(B*(f,g)) = ∂(f/(fn*gV))/∂V + ∂(g/(fn*gV))/∂n
#   A cleaner argument: the index theorem / Bendixson-Dulac.
#   Standard result: if fn*gV > 0 everywhere, then by the Dulac criterion with B=1,
#   div(f,g) = fV+gn; if this has constant sign, no limit cycles exist.
#   The more direct argument: if fn*gV>0 for all V,n (the "non-standard" sign assumptions
#   removed), one can show via the Dulac function B(V,n)=1/(fn(V,n)*gV(V,n)) that the
#   divergence of (B*f, B*g) has constant sign, precluding closed orbits. ✓
#
# 2.4.5  Extra Credit: HH equations with four limit cycles
#   The Hodgkin-Huxley (HH) model is a 4D system. For specific parameter regimes
#   (modified HH, e.g., injecting a current near bistability), numerical continuation
#   tools (AUTO, XPPAUT) can reveal coexisting limit cycles via period-doubling cascades.
#   Finding I numerically requires AUTO/XPPAUT — not done here in Python.
# ═══════════════════════════════════════════════

print("\n=== 2.4  Fun with planar dynamics ===")
print("""
2.4.1  Prove equilibrium with fV<0 is stable:
  Jacobian J = [[fV, fn], [gV, gn]]
  Tr(J) = fV + gn < 0  (fV<0 given; gn<0 by assumption)
  Det(J) = fV*gn - fn*gV = (+) - (neg*pos) = (+)+(+) > 0
  Tr<0, Det>0 → all eigenvalues have negative real parts → STABLE. ✓

2.4.2  Middle-branch equilibrium has fV>0:
  On V-nullcline (f=0): implicit diff gives slope dn/dV = -fV/fn.
  Middle branch of cubic is increasing, so dn/dV > 0.
  Since fn<0: -fV/fn>0 iff fV>0. ✓

2.4.3  Middle-branch equilibrium is a saddle when n-null slope < V-null slope:
  Slope condition: -gV/gn < -fV/fn
  Cross-multiplying (fn<0, gn<0 both flip inequalities):
    → fV*gn - gV*fn < 0  ↔  Det(J) < 0
  Det(J)<0 → one positive + one negative eigenvalue → SADDLE. ✓

2.4.4  Extra Credit: No limit cycles when fn*gV>0 everywhere:
  By the Dulac-Bendixson criterion with auxiliary function B=1/(fn*gV)>0,
  div(B*f, B*g) has constant sign over any simply-connected region,
  precluding closed orbits (limit cycles). ✓

2.4.5  Extra Credit: Four HH limit cycles:
  Requires numerical continuation (AUTO/XPPAUT). Known parameter regimes
  near period-doubling bifurcations in modified HH; not solved analytically.
""")

# ─────────────────────────────────────────────
# Figure 2.4: Illustrative phase plane for 2D system
#   Use a simple Morris-Lecar-like nullcline structure to show
#   the geometry of 2.4.1 – 2.4.3
# ─────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))
fig2.suptitle('Problem 2.4 – Planar Dynamics (Morris-Lecar–style geometry)',
              fontsize=11, color='white')

V_p = np.linspace(-80, 40, 600)

# ── 2.4.1  Stable equilibrium (fV<0 on right branch) ──
ax = axes2[0]
# V-nullcline: cubic-like  n = 0.5*(1+tanh((V+10)/18))  (right branch, fV<0)
n_vnull = 0.5*(1 + np.tanh((V_p + 10) / 18))
# n-nullcline: n = 0.5*(1+tanh((V-5)/10))  (monotone increasing, gV>0)
n_nnull = 0.5*(1 + np.tanh((V_p - 5) / 10))
ax.plot(V_p, n_vnull, color='cyan',  lw=2,   label='V-nullcline (f=0)')
ax.plot(V_p, n_nnull, color='lime',  lw=2,   label='n-nullcline (g=0)')
# Intersection: near V≈20, n≈0.87
ax.scatter([20], [0.87], color='yellow', s=100, zorder=6, label='Stable eq. (fV<0)')
ax.annotate('fV<0, gn<0\nTr<0, Det>0\n→ Stable', xy=(20,0.87),
            xytext=(5, 0.65), color='#ffff88', fontsize=7.5,
            arrowprops=dict(arrowstyle='->', color='#ffff88', lw=0.8))
ax.set_xlim(-80, 40); ax.set_ylim(-0.05, 1.05)
ax.set_xlabel('V (mV)', fontsize=9); ax.set_ylabel('n (gating var)', fontsize=9)
ax.set_title('2.4.1  Stable equilibrium (fV<0)', fontsize=10)
ax.legend(fontsize=8); clean_ax(ax)

# ── 2.4.2 & 2.4.3  Cubic nullcline + middle-branch saddle ──
ax = axes2[1]
# Cubic-shaped V-nullcline: n = A*V³ + B*V² (scaled to fit)
Vc = np.linspace(-2.5, 2.5, 600)
n_cubic = -(Vc**3)/3 + Vc        # cubic: left branch↓, middle↑, right↓
# W-nullcline: two slopes to show 2.4.3
n_wnull_less  = 0.5 * Vc         # slope < V-null middle slope (→ saddle)
n_wnull_more  = 1.5 * Vc         # slope > V-null middle slope (→ not saddle)
ax.plot(Vc, n_cubic,      color='cyan',    lw=2,   label='V-nullcline (cubic)')
ax.plot(Vc, n_wnull_less, color='#ff8844', lw=2,   ls='--', label='n-null: slope<V-null → saddle')
ax.plot(Vc, n_wnull_more, color='#44ff88', lw=2,   ls=':',  label='n-null: slope>V-null → other')
# Middle-branch saddle intersection
ax.scatter([0.75], [0.375], color='red',    s=90, marker='x', lw=2.5, zorder=7,
           label='Saddle (Det<0)')
ax.scatter([-0.75], [-0.375], color='#44ff88', s=80, zorder=7, label='Other eq.')
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
ax.set_xlabel('V', fontsize=9); ax.set_ylabel('n', fontsize=9)
ax.set_title('2.4.2–3  Middle branch & saddle', fontsize=10)
ax.legend(fontsize=7); clean_ax(ax)
ax.text(0.02, 0.03, 'Middle branch: fV>0\nSaddle iff n-null slope < V-null slope',
        transform=ax.transAxes, fontsize=7, color='#aaaaaa')

# ── 2.4.4  Dulac criterion illustration ──
ax = axes2[2]
# Show a phase portrait where fn*gV>0 → no limit cycles
V_d = np.linspace(-1.5, 1.5, 20)
n_d = np.linspace(-1.5, 1.5, 20)
VV, NN = np.meshgrid(V_d, n_d)
# Simple system: dV/dt = -V + n (fn=1>0), dn/dt = V - 2n (gV=1>0) → fn*gV>0
dV = -VV + NN
dN =  VV - 2*NN
speed = np.sqrt(dV**2 + dN**2) + 1e-10
ax.streamplot(V_d, n_d, dV, dN, color='#5588ff', linewidth=0.9,
              density=1.0, arrowsize=0.9)
ax.scatter([0], [0], color='yellow', s=100, zorder=6, label='Stable eq. point')
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('V', fontsize=9); ax.set_ylabel('n', fontsize=9)
ax.set_title('2.4.4  fn·gV>0 → no limit cycles', fontsize=10)
ax.legend(fontsize=8); clean_ax(ax)
ax.text(0.02, 0.03, 'Dulac-Bendixson: B=1/(fn·gV)>0\ndiv(B·F) const. sign → no closed orbits',
        transform=ax.transAxes, fontsize=7, color='#aaaaaa')

plt.tight_layout(pad=0.8, h_pad=1.0, w_pad=0.8)
plt.show()
