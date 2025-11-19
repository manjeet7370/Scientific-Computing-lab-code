from math import prod

x = [5, 7, 11, 13, 17]
y = [150, 392, 1452, 2366, 5202]
X_eval = 9
n = len(x)

print("=== Given Data Points ===")
for xi, yi in zip(x, y):
    print(f"x = {xi}, f(x) = {yi}")
print()

print(f"=== Lagrange Interpolation Table for X = {X_eval} ===")
print(f"{'i':<3}{'x_i':<6}{'f(x_i)':<10}{'Num Product':<15}{'Den Product':<15}{'L_i(X)':<15}{'Contribution':<15}")

f_lagrange = 0
for i in range(n):
    numer_factors = [X_eval - x[j] for j in range(n) if j != i]
    denom_factors = [x[i] - x[j] for j in range(n) if j != i]
    numerator = prod(numer_factors)
    denominator = prod(denom_factors)
    L_i = numerator / denominator
    contribution = y[i] * L_i
    f_lagrange += contribution

    print(f"{i:<3}{x[i]:<6}{y[i]:<10}{numerator:<15.6f}{denominator:<15.6f}{L_i:<15.6f}{contribution:<15.6f}")

print(f"\nFinal f({X_eval}) using Lagrange = {f_lagrange}")
