import numpy as np

# --- 1. Problem Parameters ---
L = 5.0  # Length of the domain (0 <= x <= 5)
T = 0.4  # Final time
C = 5.0  # Coefficient in the PDE: du/dt = C * (d^2u/dx^2)

# --- 2. Discretization Parameters ---
h = 1.0  # Spatial step size
k = 0.05 # Time step size (chosen such that lambda <= 0.5)
Nx = int(L / h)  # Number of spatial segments (5/1 = 5)
Nt = int(T / k) + 1  # Number of time levels (0.4/0.05 = 8 + 1 = 9)

# Stability parameter (lambda)
lambda_val = (C * k) / (h**2)

print(f"--- Discretization Details ---")
print(f"Spatial steps (h): {h}, Number of points (Nx+1): {Nx + 1}")
print(f"Time steps (k): {k}, Number of levels (Nt): {Nt}")
print(f"Stability parameter (lambda): {lambda_val}\n") # Output: 0.25

# --- 3. Initialize Grid ---
x = np.linspace(0, L, Nx + 1) # x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
u_current = np.zeros(Nx + 1)
u_next = np.zeros(Nx + 1)

# --- 4. Apply Initial Condition (t=0, j=0) ---

for i in range(Nx + 1):
    xi = x[i]
    if xi <= 3.0:
        # u(x, 0) = 50x for 0 <= x <= 3
        u_current[i] = 50 * xi
    else:
        # u(x, 0) = 60 for 3 < x <= 5
        u_current[i] = 60

# Ensure boundary points adhere to BCs if necessary, although the IC covers it here
# u_current[0] = 60  (From 50*0 = 0, this needs correction based on BC)
# u_current[Nx] = 60 (From IC, this is 60)

# Correcting u_current[0] and u_current[Nx] with the actual Boundary Conditions
u_current[0] = 60.0  # u(0, t) = 60
u_current[Nx] = 60.0 # u(5, t) = 60

print("Initial Condition (t=0) after BC application:")
print(u_current)
print("-" * 30)

# --- 5. Implement Schmidt's Method (Explicit Finite Difference) ---
constant_term = 1 - 2 * lambda_val # 1 - 2*0.25 = 0.5

for j in range(Nt - 1):  # Iterate through time steps
    t = (j + 1) * k  # Time at the next level

    # Apply Boundary Conditions for the next level
    u_next[0] = 60.0  # u(0, t) = 60
    u_next[Nx] = 60.0  # u(5, t) = 60

    # Calculate interior points (i=1 to Nx-1)
    for i in range(1, Nx):
        # Schmidt's formula: u_{i, j+1} = lambda*u_{i-1, j} + (1 - 2*lambda)*u_{i, j} + lambda*u_{i+1, j}
        u_next[i] = lambda_val * u_current[i - 1] + \
                    constant_term * u_current[i] + \
                    lambda_val * u_current[i + 1]

    # Update current solution for the next time step
    u_current = np.copy(u_next)

    print(f"Solution at t={t:.3f} (j={j+1}):")
    print(u_current)

print("\n--- Final Solution at T=0.4 ---")
print(f"x-values: {x}")
print(f"u(x, 0.4): {u_current}")