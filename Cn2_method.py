import numpy as np
from scipy.linalg import solve_banded

# --- 1. Problem Parameters ---
L = 1.0  # Length of the domain (0 <= x <= 1)
T = 0.05 # Final time
C = 1.0  # Coefficient (Implicit in the PDE: du/dt = 1 * d^2u/dx^2)

# --- 2. Discretization Parameters ---
h = 0.1  # Spatial step size
k = 0.01 # Time step size
Nx = int(L / h)  # Number of spatial segments (1/0.1 = 10)
Nt = int(T / k) + 1  # Number of time levels (0.05/0.01 = 5 + 1 = 6)

# Stability/Discretization parameter (lambda)
lambda_val = (C * k) / (h**2) # lambda = 1.0 * 0.01 / 0.1^2 = 1.0

print(f"--- Discretization Details ---")
print(f"Spatial steps (h): {h}, Number of points (Nx+1): {Nx + 1}")
print(f"Time steps (k): {k}, Number of levels (Nt): {Nt}")
print(f"Discretization parameter (lambda): {lambda_val}\n")

# --- 3. Initialize Grid and System Matrices ---
x = np.linspace(0, L, Nx + 1)
u_current = np.zeros(Nx + 1)

# The implicit system A*u^{j+1} = b is solved for interior points (i=1 to Nx-1)
N_interior = Nx - 1

# Define the constants for the tridiagonal system A*u^{j+1} = b
# A matrix coefficients (for the left-hand side, u^{j+1})
a = -lambda_val              # Sub-diagonal
b = 2.0 + 2.0 * lambda_val   # Main diagonal
c = -lambda_val              # Super-diagonal

# B matrix coefficients (for the right-hand side, u^j)
alpha = lambda_val
beta = 2.0 - 2.0 * lambda_val
gamma = lambda_val

# Construct the tridiagonal matrix A for the interior points
# The structure needed for solve_banded is (2, N, M), where 2 is the bandwidth (1 below, 1 above)
# [a, b, c, ...]
A_banded = np.zeros((3, N_interior))
A_banded[0, 1:] = c  # Super-diagonal (c)
A_banded[1, :] = b   # Main diagonal (b)
A_banded[2, :-1] = a # Sub-diagonal (a)

# --- 4. Apply Initial Condition (t=0, j=0) ---
# Initial condition: u(x, 0) = 100 * (x - x^2)
for i in range(Nx + 1):
    u_current[i] = 100 * (x[i] - x[i]**2)

# Apply Boundary Conditions
u_current[0] = 0.0  # u(0, t) = 0
u_current[Nx] = 0.0  # u(1, t) = 0

print("Initial Condition (t=0):")
print(np.round(u_current, 4))
print("-" * 30)

# --- 5. Implement Crank-Nicolson Method (using TDMA/solve_banded) ---

for j in range(Nt - 1):
    t = (j + 1) * k

    # 1. Calculate the right-hand side vector (b_vector) for the system A*u^{j+1} = b
    # b_i = alpha*u_{i-1, j} + beta*u_{i, j} + gamma*u_{i+1, j}
    b_vector = np.zeros(N_interior)

    for i in range(1, Nx):
        if i == 1:
            # i=1 (first interior point): includes u_0 which is a Boundary Condition
            b_vector[i-1] = alpha * u_current[i - 1] + \
                            beta * u_current[i] + \
                            gamma * u_current[i + 1]
            # Since u_0=0, the equation simplifies, but we still have to account for it 
            # by moving the known term (lambda*u_0) to the RHS.
            # R.H.S (TDMA) = (alpha*u_{i-1} + beta*u_i + gamma*u_{i+1}) + lambda*u_{i-1, j+1} (known BC)
            # Since u_{0, j+1}=0, no further change needed, as lambda*u_{0, j+1} = 0.

        elif i == Nx - 1:
            # i=Nx-1 (last interior point): includes u_Nx which is a Boundary Condition
            b_vector[i-1] = alpha * u_current[i - 1] + \
                            beta * u_current[i] + \
                            gamma * u_current[i + 1]
            # Similarly, u_{Nx, j+1}=0, so no further change needed.

        else:
            # Interior points
            b_vector[i-1] = alpha * u_current[i - 1] + \
                            beta * u_current[i] + \
                            gamma * u_current[i + 1]

    # 2. Solve the linear system A*u_{interior}^{j+1} = b_vector
    u_interior_next = solve_banded((1, 1), A_banded, b_vector)

    # 3. Update the full solution vector u_current
    u_current[1:Nx] = u_interior_next
    # Boundary points remain u_current[0]=0.0 and u_current[Nx]=0.0

    print(f"Solution at t={t:.3f} (j={j+1}):")
    print(np.round(u_current, 4))

print("\n--- Final Solution at T=0.05 ---")
print(f"x-values: {x}")
print(f"u(x, 0.05): {np.round(u_current, 4)}")