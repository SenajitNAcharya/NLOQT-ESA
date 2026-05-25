import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# =============================================================================
# MASTER SETTINGS & PUBLICATION FORMATTING
# =============================================================================
plt.rcParams.update({
    'font.size': 12, 
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.titlesize': 18
})

print("===================================================================")
print(" SPDC QUANTUM CORRELATION SIMULATION SUITE (ADVANCED PHYSICS)")
print(" Material: Congruent Lithium Niobate (LiNbO3) - Type 0 Phase Matching")
print("===================================================================\n")

# =============================================================================
# PHYSICAL CONSTANTS & LITHIUM NIOBATE DISPERSION (SELLMEIER EQUATIONS)
# =============================================================================
print("--> Initializing Sellmeier Dispersion Models...")

def ne_LiNbO3(lmbda_m):
    """Sellmeier equation for extraordinary refractive index of congruent LiNbO3."""
    l_um = lmbda_m * 1e6
    return np.sqrt(5.35583 + 0.100473 / (l_um**2 - 0.042798) - 0.053625 * l_um**2)

c = 3e8
lmbda_p0 = 405e-9  # Pump wavelength (405 nm)
lmbda_s0 = 810e-9  # Signal central wavelength (810 nm)
lmbda_i0 = 810e-9  # Idler central wavelength (810 nm)

# Phase Mismatch Calculation for Nominal Poling Period
kp_0 = 2 * np.pi * ne_LiNbO3(lmbda_p0) / lmbda_p0
ks_0 = 2 * np.pi * ne_LiNbO3(lmbda_s0) / lmbda_s0
ki_0 = 2 * np.pi * ne_LiNbO3(lmbda_i0) / lmbda_i0

Lambda_QPM = 2 * np.pi / (kp_0 - ks_0 - ki_0) # Required Poling Period

# =============================================================================
# MODULE 1: SPECTRAL CORRELATIONS (1D Longitudinal Structuring)
# =============================================================================
print("--> MODULE 1: Simulating Spectral Correlations (1D)...")

L = 10e-3               # Crystal length (10 mm)
N_z = 20000             # Spatial grid points (z-axis)
z = np.linspace(-L/2, L/2, N_z)

# 1.1 Generate 1D chi^(2) Profiles
profiles_1D = {
    'Standard QPM': np.sign(np.cos(2 * np.pi * z / Lambda_QPM)),
    'Gaussian Apodized': np.sign(np.cos(2 * np.pi * z / Lambda_QPM)) * np.exp(-(z**2) / ((L/4)**2))
}

# 1.2 Compute 2D Joint Spectral Intensity (JSI) using Physical Dispersion
dw = np.linspace(-2e12, 2e12, 250) # Frequency detuning grid in rad/s
DW_s, DW_i = np.meshgrid(dw, dw)

# Calculate physical momentum vectors across the frequency grid
w_s0 = 2 * np.pi * c / lmbda_s0
w_i0 = 2 * np.pi * c / lmbda_i0
w_p = (w_s0 + DW_s) + (w_i0 + DW_i) # Strict Energy conservation

K_p = w_p * ne_LiNbO3(2 * np.pi * c / w_p) / c
K_s = (w_s0 + DW_s) * ne_LiNbO3(2 * np.pi * c / (w_s0 + DW_s)) / c
K_i = (w_i0 + DW_i) * ne_LiNbO3(2 * np.pi * c / (w_i0 + DW_i)) / c

DK_2D = K_p - K_s - K_i
Pump_Intensity = np.exp(-((DW_s + DW_i)**2) / (5e11**2)) # 500 GHz pump bandwidth

def compute_jsi(chi_z, z, DK_grid):
    dk_flat = DK_grid.flatten()
    batch_size = 1000  # Memory-safe batching
    pmf_flat = np.zeros_like(dk_flat)
    for i in range(0, len(dk_flat), batch_size):
        batch = dk_flat[i:i+batch_size]
        integrand = chi_z[None, :] * np.exp(1j * batch[:, None] * z[None, :])
        pmf_flat[i:i+batch_size] = np.abs(np.trapezoid(integrand, x=z, axis=1))**2
    jsi = Pump_Intensity * (pmf_flat.reshape(DK_grid.shape) / np.max(pmf_flat))
    return jsi / np.max(jsi)

jsi_qpm = compute_jsi(profiles_1D['Standard QPM'], z, DK_2D)
jsi_apodized = compute_jsi(profiles_1D['Gaussian Apodized'], z, DK_2D)

# =============================================================================
# MODULE 2: SPATIAL CORRELATIONS (2D Transverse Structuring)
# =============================================================================
print("--> MODULE 2: Simulating Spatial Correlations (2D Transverse)...")

xy_max = 50e-6
N_xy = 300
x = np.linspace(-xy_max, xy_max, N_xy)
y = np.linspace(-xy_max, xy_max, N_xy)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Phi = np.arctan2(Y, X)

# Model Spatially Varying Nonlinear Coefficient (Fork Grating)
w_pump = 25e-6
pump_spatial = np.exp(-(R**2) / (w_pump**2))
l_charge = 1
Lambda_x = 10e-6
k_x = 2 * np.pi / Lambda_x
chi2_2d = np.sign(np.cos(k_x * X + l_charge * Phi))

# Compute Spatial Photon Correlation Pattern
biphoton_spatial = pump_spatial * chi2_2d
spatial_correlation_pattern = np.abs(biphoton_spatial)**2

# Compute OAM Spectrum (Projection onto spiral harmonics)
m_values = np.arange(-5, 6)
oam_spectrum = np.zeros_like(m_values, dtype=float)
for idx, m in enumerate(m_values):
    overlap = np.sum(biphoton_spatial * np.exp(-1j * m * Phi) * R)
    oam_spectrum[idx] = np.abs(overlap)**2
oam_spectrum /= np.sum(oam_spectrum)

# =============================================================================
# MODULE 3: QUANTUM VALIDATION (Schmidt Decomposition)
# =============================================================================
print("--> MODULE 3: Computing Quantum Entanglement Metrics...")

jsa_apodized = np.sqrt(jsi_apodized)
U, S_vals, Vh = svd(jsa_apodized)
schmidt_probs = (S_vals / np.linalg.norm(S_vals))**2
schmidt_number = 1.0 / np.sum(schmidt_probs**2)
entropy = -np.sum(schmidt_probs[schmidt_probs > 1e-10] * np.log2(schmidt_probs[schmidt_probs > 1e-10]))

# =============================================================================
# PLOTTING & SAVING VISUALIZATIONS
# =============================================================================
print("--> Generating and saving plots...")

# Convert detuning to THz for cleaner plot axes
DW_s_THz = DW_s / (2 * np.pi * 1e12)
DW_i_THz = DW_i / (2 * np.pi * 1e12)

# PLOT 1: Spectral Control
fig1 = plt.figure(figsize=(15, 5))
ax1 = plt.subplot(131)
c1 = ax1.contourf(DW_s_THz, DW_i_THz, jsi_qpm, levels=50, cmap='magma')
ax1.set_title("JSI: Standard QPM (LiNbO$_3$)\n(High Spectral Correlation)", fontsize=13)
ax1.set_xlabel(r"Signal Detuning $\Delta\nu_s$ (THz)")
ax1.set_ylabel(r"Idler Detuning $\Delta\nu_i$ (THz)")

ax2 = plt.subplot(132)
c2 = ax2.contourf(DW_s_THz, DW_i_THz, jsi_apodized, levels=50, cmap='magma')
ax2.set_title("JSI: Apodized Crystal\n(Decoupled Spectral Correlation)", fontsize=13)
ax2.set_xlabel(r"Signal Detuning $\Delta\nu_s$ (THz)")

ax3 = plt.subplot(133)
ax3.bar(range(1, 11), schmidt_probs[:10], color='#2980b9', edgecolor='black')
ax3.set_title(f"Schmidt Modes (Apodized)\nK = {schmidt_number:.2f}, S = {entropy:.2f} bits", fontsize=13)
ax3.set_xlabel("Schmidt Mode $n$")
ax3.set_ylabel(r"Probability $\lambda_n^2$")
ax3.set_xlim(0, 10)

plt.tight_layout()
plt.savefig('Result_1_Spectral_Correlations.png', dpi=300)
plt.close(fig1)

# PLOT 2: Spatial Control 
fig2 = plt.figure(figsize=(15, 5))
ax4 = plt.subplot(131)
ax4.imshow(chi2_2d, extent=[-50, 50, -50, 50], cmap='gray', origin='lower')
ax4.set_title(r"Spatially Varying $\chi^{(2)}(x,y)$" + "\n(Fork Grating $l=1$)", fontsize=13)
ax4.set_xlabel(r"$x$ ($\mu$m)")
ax4.set_ylabel(r"$y$ ($\mu$m)")

ax5 = plt.subplot(132)
ax5.imshow(spatial_correlation_pattern, extent=[-50, 50, -50, 50], cmap='inferno', origin='lower')
ax5.set_title("Photon Spatial Correlation\n(Near-Field Pattern)", fontsize=13)
ax5.set_xlabel(r"$x$ ($\mu$m)")

ax6 = plt.subplot(133)
ax6.bar(m_values, oam_spectrum, color='#8e44ad', edgecolor='black')
ax6.set_title(r"OAM Correlation Spectrum", fontsize=13)
ax6.set_xlabel(r"Topological Charge $m$")
ax6.set_ylabel("Probability")
ax6.set_xticks(m_values)

plt.tight_layout()
plt.savefig('Result_2_Spatial_Correlations.png', dpi=300)
plt.close(fig2)

print("\nSUCCESS! All simulations completed.")
print("Saved:")
print("  - Result_1_Spectral_Correlations.png")
print("  - Result_2_Spatial_Correlations.png")
