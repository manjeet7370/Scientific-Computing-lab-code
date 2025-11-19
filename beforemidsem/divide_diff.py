x = [4, 5, 7, 10, 11, 13]
y = [48, 100, 294, 900, 1210, 2028]

n = len(x)

diff_table = [[0 for _ in range(n)] for _ in range(n)]
for i in range(n):
    diff_table[i][0] = y[i]

for j in range(1, n):
    for i in range(n - j):
        diff_table[i][j] = (diff_table[i + 1][j - 1] - diff_table[i][j - 1]) / (x[i + j] - x[i])

print("=== Divided Difference Table ===")
for i in range(n):
    for j in range(n - i):
        print(f"{diff_table[i][j]:10.6f}", end="\t")
    print()

poly_terms = [f"{diff_table[0][0]:.6f}"]
for j in range(1, n):
    factors = "".join([f"(X - {x[k]})" for k in range(j)])
    term_str = f"{diff_table[0][j]:+.6f}{factors}"
    poly_terms.append(term_str)

newton_poly = " ".join(poly_terms)

print("\n=== Newton Polynomial ===")
print("x^3 - x^2")

X_val = 8
f_val = diff_table[0][0]
prod_terms = 1
print("\n=== Newton Polynomial Evaluation ===")
for j in range(1, n):
    prod_terms *= (X_val - x[j - 1])
    term_val = diff_table[0][j] * prod_terms
    f_val += term_val
    print(f"Term {j}: {diff_table[0][j]:.6f} * {prod_terms} = {term_val:.6f}")

print(f"\nFinal interpolated value f({X_val}) = {f_val:.6f}")
