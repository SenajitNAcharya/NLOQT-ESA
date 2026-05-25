import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import time

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
print(" SPDC QUANTUM CORRELATION SIMULATION SUITE")
print(" Modulating χ⁽²⁾ for Spatial and Spectral Photon Control")
print("===================================================================\n")

# =============================================================================
# MODULE 1: SPECTRAL CORRELATIONS (1D Longitudinal Structuring)
# Validates old work: PMF, JSI, Apodization, and Fabrication Errors
# =============================================================================
print("--> MODULE 1: Simulating Spectral Correlations (1D)...")

L = 10e-3               # Crystal length (10 mm)
Lambda = 10e-6          # Poling period (10 um)
N_z = 20000             # Spatial grid points (z-axis)
z = np.linspace(-L/2, L/2, N_z)
dk_0 = 2 * np.pi / Lambda
dk_range = np.linspace(dk_0 - 3000, dk_0 + 3000, 800)

# 1.1 Generate 1D χ⁽²⁾ Profiles
profiles_1D = {
    'Uniform': np.ones_like(z),
    'Standard QPM': np.sign(np.cos(2 * np.pi * z / Lambda)),
    'Gaussian Apodized': np.sign(np.cos(2 * np.pi * z / Lambda)) * np.exp(-(z**2) / ((L/4)**2)),
    'Random Duty Cycle (Error)': np.sign(np.cos(2 * np.pi * z / Lambda + np.convolve(0.4 * np.random.randn(len(z)), np.ones(50)/50, mode='same')))
}

# 1.2 Compute PMF (Memory-safe integration)
def compute_pmf_1d(chi_z, z, dk_array):
    integrand = chi_z[None, :] * np.exp(1j * dk_array[:, None] * z[None, :])
    pmf_complex = np.trapezoid(integrand, x=z, axis=1) # Fixed for NumPy 2.0+
    pmf = np.abs(pmf_complex)**2
    return pmf / np.max(pmf)

pmf_results = {name: compute_pmf_1d(chi, z, dk_range) for name, chi in profiles_1D.items()}

# 1.3 Compute 2D Joint Spectral Intensity (JSI)
dw = np.linspace(-1500, 1500, 250)
DW_s, DW_i = np.meshgrid(dw, dw)
DK_2D = dk_0 + 1.0 * DW_s - 0.8 * DW_i  # Simulated dispersion
Pump_Intensity = np.exp(-((DW_s + DW_i)**2) / (500**2))

def compute_jsi(chi_z, z, DK_grid):
    dk_flat = DK_grid.flatten()
    # Batch processing to prevent MemoryError (RAM crash)
    batch_size = 1000
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
# Actual Problem Statement: Spatial Varying Coefficient & Photon Patterns
# =============================================================================
print("--> MODULE 2: Simulating Spatial Correlations (2D Transverse)...")

# 2.1 Set up 2D Transverse Grid (x, y)
xy_max = 50e-6
N_xy = 300
x = np.linspace(-xy_max, xy_max, N_xy)
y = np.linspace(-xy_max, xy_max, N_xy)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Phi = np.arctan2(Y, X)

# 2.2 Model Spatially Varying Nonlinear Coefficient (Fork Grating)
w_pump = 25e-6
pump_spatial = np.exp(-(R**2) / (w_pump**2))
l_charge = 1         # Topological charge
Lambda_x = 10e-6     # Grating period in transverse plane
k_x = 2 * np.pi / Lambda_x

# The Fork Grating Equation: chi2(x,y)
chi2_2d = np.sign(np.cos(k_x * X + l_charge * Phi))

# 2.3 Compute Spatial Photon Correlation Pattern (Near-field biphoton amplitude)
# In the near-field collinear regime, spatial correlation mimics the pump * nonlinearity
biphoton_spatial = pump_spatial * chi2_2d
spatial_correlation_pattern = np.abs(biphoton_spatial)**2

# 2.4 Compute OAM Spectrum (Projection onto spiral harmonics)
m_values = np.arange(-5, 6)
oam_spectrum = np.zeros_like(m_values, dtype=float)

for idx, m in enumerate(m_values):
    overlap = np.sum(biphoton_spatial * np.exp(-1j * m * Phi) * R) # Area element R dR dPhi
    oam_spectrum[idx] = np.abs(overlap)**2
oam_spectrum /= np.sum(oam_spectrum) # Normalize

# =============================================================================
# MODULE 3: QUANTUM VALIDATION (Schmidt Decomposition)
# =============================================================================
print("--> MODULE 3: Computing Quantum Entanglement Metrics...")

# Perform Singular Value Decomposition on the Apodized JSA (sqrt of JSI)
jsa_apodized = np.sqrt(jsi_apodized)
U, S_vals, Vh = svd(jsa_apodized)
schmidt_probs = (S_vals / np.linalg.norm(S_vals))**2

# Calculate Metrics
schmidt_number = 1.0 / np.sum(schmidt_probs**2)
entropy = -np.sum(schmidt_probs[schmidt_probs > 1e-10] * np.log2(schmidt_probs[schmidt_probs > 1e-10]))

# =============================================================================
# PLOTTING & SAVING VISUALIZATIONS
# =============================================================================
print("--> Generating and saving plots...")

# PLOT 1: Spectral Control (JSI & Schmidt)
fig1 = plt.figure(figsize=(15, 5))
ax1 = plt.subplot(131)
c1 = ax1.contourf(DW_s, DW_i, jsi_qpm, levels=50, cmap='magma')
ax1.set_title("JSI: Standard QPM\n(High Spectral Correlation)", fontsize=13)
ax1.set_xlabel(r"Signal Detuning $\Delta\omega_s$")
ax1.set_ylabel(r"Idler Detuning $\Delta\omega_i$")

ax2 = plt.subplot(132)
c2 = ax2.contourf(DW_s, DW_i, jsi_apodized, levels=50, cmap='magma')
ax2.set_title("JSI: Apodized Crystal\n(Decoupled Spectral Correlation)", fontsize=13)
ax2.set_xlabel(r"Signal Detuning $\Delta\omega_s$")

ax3 = plt.subplot(133)
ax3.bar(range(1, 11), schmidt_probs[:10], color='#2980b9', edgecolor='black')
ax3.set_title(f"Schmidt Modes (Apodized)\nK = {schmidt_number:.2f}, S = {entropy:.2f} bits", fontsize=13)
ax3.set_xlabel("Schmidt Mode $n$")
ax3.set_ylabel(r"Probability $\lambda_n^2$")
ax3.set_xlim(0, 10)

plt.tight_layout()
plt.savefig('Result_1_Spectral_Correlations.png', dpi=300)
plt.close(fig1)

# PLOT 2: Spatial Control (The Image Problem Statement)
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
print("  - Result_1_Spectral_Correlations.png (Validates old 1D work)")
print("  - Result_2_Spatial_Correlations.png  (Solves new 2D spatial problem)")