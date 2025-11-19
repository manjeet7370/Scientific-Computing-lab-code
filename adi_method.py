import numpy as np
from scipy.linalg import solve_banded

# --- 1. Problem Parameters ---
L_x = 0.75  # Length of the x-domain
L_y = 0.75  # Length of the y-domain
T_final = 0.04 # Final time to run the simulation
C = 1.0     # Coefficient (u_t = C * (u_xx + u_yy))

# --- 2. Discretization Parameters ---
h = 0.25    # Spatial step size (h = dx = dy)
k = 0.01    # Total time step size (k)

# Derived parameters
Nx = int(L_x / h)  # Number of steps = 3
Ny = int(L_y / h)  # Number of steps = 3
Nt = int(T_final / k) # Number of time iterations = 4

lambda_val = (C * k) / (h**2) # 0.16
lambda_half = lambda_val / 2.0 # 0.08

N_interior_x = Nx - 1 # 2 interior points in x (i=1, 2)
N_interior_y = Ny - 1 # 2 interior points in y (j=1, 2)

print(f"--- ADI Discretization Details ---")
print(f"Spatial steps (h): {h}, Grid: {(Nx+1)}x{(Ny+1)}")
print(f"Time step (k): {k}, lambda: {lambda_val}, lambda/2: {lambda_half}")
print("-" * 40)

# --- 3. Initialize Grid ---
# u_grid stores the solution u(x_i, y_j, t_n)
u_grid = np.zeros((Nx + 1, Ny + 1))
# u_star will store the solution at the intermediate half-step (n+1/2)
u_star = np.zeros((Nx + 1, Ny + 1))
# u_next will store the solution at the full next step (n+1)
u_next = np.zeros((Nx + 1, Ny + 1))

# --- 4. Apply Initial Condition (t=0, n=0) ---
x_pts = np.linspace(0, L_x, Nx + 1)
y_pts = np.linspace(0, L_y, Ny + 1)

# Initial condition: u(x, y, 0) = 10xy
for i in range(Nx + 1):
    for j in range(Ny + 1):
        u_grid[i, j] = 10 * x_pts[i] * y_pts[j]

# Apply Boundary Conditions (All boundaries are zero)
u_grid[0, :] = 0.0
u_grid[Nx, :] = 0.0
u_grid[:, 0] = 0.0
u_grid[:, Ny] = 0.0

print(f"Initial Condition (t=0):\n{np.round(u_grid, 4)}")
print("-" * 40)

# --- 5. TDMA Setup (Constant Matrix A_x and A_y) ---

# A_x (LHS of x-step): Implicit in x
ax = -lambda_half
bx = 1.0 + 2.0 * lambda_half
cx = -lambda_half

A_x_banded = np.zeros((3, N_interior_x))
A_x_banded[0, 1:] = cx    # Super-diagonal
A_x_banded[1, :] = bx     # Main diagonal
A_x_banded[2, :-1] = ax   # Sub-diagonal

# --- 6. ADI Time Stepping Loop ---

for n in range(Nt):
    # --- STEP 1: Implicit in x, Explicit in y (n -> n+1/2) ---
    
    # Boundary conditions for u_star (same as u_grid for the boundaries)
    u_star[0, :] = 0.0
    u_star[Nx, :] = 0.0
    u_star[:, 0] = 0.0
    u_star[:, Ny] = 0.0

    for j in range(1, Ny): # Loop over each y-line (j)
        # 1. Calculate the Right-Hand Side (RHS) vector for this y-line (b_x_vector)
        # RHS_i = lambda/2 * u_{i, j-1}^n + (1 - lambda) * u_{i, j}^n + lambda/2 * u_{i, j+1}^n
        b_x_vector = np.zeros(N_interior_x)
        
        for i in range(1, Nx):
            # Calculate the RHS using u^n values
            rhs_val = lambda_half * u_grid[i, j - 1] + \
                      (1.0 - lambda_val) * u_grid[i, j] + \
                      lambda_half * u_grid[i, j + 1]
            
            # Since BCs are 0, no modification to RHS is needed
            b_x_vector[i-1] = rhs_val

        # 2. Solve the linear system A_x * u_star_interior = b_x_vector
        u_star_interior = solve_banded((1, 1), A_x_banded, b_x_vector)

        # 3. Update the u_star grid for this y-line (j)
        u_star[1:Nx, j] = u_star_interior
        
    # --- STEP 2: Implicit in y, Explicit in x (n+1/2 -> n+1) ---
    
    # Boundary conditions for u_next (same as u_star for the boundaries)
    u_next[0, :] = 0.0
    u_next[Nx, :] = 0.0
    u_next[:, 0] = 0.0
    u_next[:, Ny] = 0.0

    # A_y (LHS of y-step): The matrix is structurally the same as A_x (tridiagonal)
    # The TDMA matrix is constant across all columns (x-lines), identical to A_x_banded.
    
    for i in range(1, Nx): # Loop over each x-line (i)
        # 1. Calculate the Right-Hand Side (RHS) vector for this x-line (b_y_vector)
        # RHS_j = lambda/2 * u_{i-1, j}^* + (1 - lambda) * u_{i, j}^* + lambda/2 * u_{i+1, j}^*
        b_y_vector = np.zeros(N_interior_y)
        
        for j in range(1, Ny):
            # Calculate the RHS using u_star values
            rhs_val = lambda_half * u_star[i - 1, j] + \
                      (1.0 - lambda_val) * u_star[i, j] + \
                      lambda_half * u_star[i + 1, j]
            
            # Since BCs are 0, no modification to RHS is needed
            b_y_vector[j-1] = rhs_val

        # 2. Solve the linear system A_y * u_next_interior = b_y_vector
        # Note: We are solving along the y-direction, so the resulting vector is for column i.
        u_next_interior = solve_banded((1, 1), A_x_banded, b_y_vector) # Use A_x matrix structure

        # 3. Update the u_next grid for this x-line (i)
        u_next[i, 1:Ny] = u_next_interior
        
    # Update u_grid for the next iteration
    u_grid = np.copy(u_next)
    
    current_time = (n + 1) * k
    print(f"Solution at t={current_time:.4f} (Level {n+1}):")
    # Print only the interior 3x3 grid for brevity
    print(np.round(u_grid[1:Nx, 1:Ny], 4))

print("\n--- Final Solution at T=0.04 (Interior Points) ---")
# The final solution is u_grid
print(np.round(u_grid[1:Nx, 1:Ny], 4))