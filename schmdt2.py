import numpy as np
from scipy.linalg import solve_banded

# --- 1. Problem Parameters ---
L = 1.0     # Length of the domain
T = 2 * (1/36) # Final time for two levels
C = 1.0     # Coefficient in the PDE (u_t = C * u_xx)

# --- 2. Discretization Parameters ---
h = 1/4     # Spatial step size
k = 1/36    # Time step size
Nx = int(L / h)  # Number of spatial segments (1 / 0.25 = 4)
Nt = int(T / k) + 1  # Number of time levels (2 + 1 = 3)

lambda_val = (C * k) / (h**2) # 4/9

# Create grid points
x = np.linspace(0, L, Nx + 1) # x = [0.0, 0.25, 0.5, 0.75, 1.0]

# Initial Condition: u(x, 0) = sin(pi*x)
u_init = np.sin(np.pi * x)

# Apply Boundary Conditions: u(0, t) = 0, u(1, t) = 0
u_init[0] = 0.0
u_init[Nx] = 0.0

print(f"--- Shared Discretization Details ---")
print(f"h = {h:.4f}, k = {k:.4f}, lambda = {lambda_val:.4f} (4/9)")
print(f"Grid points (Nx+1): {Nx + 1}")
print(f"Initial Condition (t=0): {np.round(u_init, 4)}\n")
## --- 2. Explicit Scheme (Schmidt's Method) ---
u_exp = np.copy(u_init)
u_next = np.zeros(Nx + 1)

# Pre-calculate constants
const_middle = 1.0 - 2.0 * lambda_val # 1 - 8/9 = 1/9

print("--- 🌟 Explicit Scheme (Schmidt's Method) ---")

for j in range(2): # Calculate for j=0 (t=k) and j=1 (t=2k)
    t = (j + 1) * k

    # Apply Boundary Conditions for the next level
    u_next[0] = 0.0
    u_next[Nx] = 0.0

    # Calculate interior points (i=1 to Nx-1)
    for i in range(1, Nx):
        u_next[i] = lambda_val * u_exp[i - 1] + \
                    const_middle * u_exp[i] + \
                    lambda_val * u_exp[i + 1]

    u_exp = np.copy(u_next)

    print(f"Solution at t={t:.4f} (Level {j+1}): {np.round(u_exp, 4)}")

print("\n")