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
## --- 3. Implicit Scheme (Crank-Nicolson Method) ---
u_cn = np.copy(u_init)
N_interior = Nx - 1 # 3 interior points

# Define constants for the system
# LHS (A matrix) coefficients
a = -lambda_val              # Sub-diagonal
b = 2.0 + 2.0 * lambda_val   # Main diagonal
c = -lambda_val              # Super-diagonal

# RHS (b vector) coefficients
alpha = lambda_val
beta = 2.0 - 2.0 * lambda_val
gamma = lambda_val

# Construct the Tridiagonal Matrix A (constant for all steps since lambda is constant)
A_banded = np.zeros((3, N_interior))
A_banded[0, 1:] = c    # Super-diagonal
A_banded[1, :] = b     # Main diagonal
A_banded[2, :-1] = a   # Sub-diagonal

print("--- 🌊 Implicit Scheme (Crank-Nicolson Method) ---")

for j in range(2): # Calculate for j=0 (t=k) and j=1 (t=2k)
    t = (j + 1) * k
    b_vector = np.zeros(N_interior)

    # 1. Calculate the right-hand side vector (b_vector) based on u^j
    for i in range(1, Nx):
        # b_i = alpha*u_{i-1, j} + beta*u_{i, j} + gamma*u_{i+1, j}
        b_vector[i-1] = alpha * u_cn[i - 1] + \
                        beta * u_cn[i] + \
                        gamma * u_cn[i + 1]
    
    # Boundary conditions are homogeneous (u_0=0, u_N=0), so no adjustment needed to b_vector.
    
    # 2. Solve the linear system A*u_{interior}^{j+1} = b_vector
    u_interior_next = solve_banded((1, 1), A_banded, b_vector)

    # 3. Update the full solution vector u_cn
    u_cn[1:Nx] = u_interior_next
    # Boundary points remain u_cn[0]=0.0 and u_cn[Nx]=0.0

    print(f"Solution at t={t:.4f} (Level {j+1}): {np.round(u_cn, 4)}")