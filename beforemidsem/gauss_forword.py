def gauss_forward_interp(x_list, y_list, target):
    """
    Gauss Forward Interpolation (restructured version)
    """
    n = len(x_list)
    step = x_list[1] - x_list[0]   

    table = [[0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = y_list[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]

    origin_index = 1
    x0 = x_list[origin_index]
    u = (target - x0) / step

    approx = table[origin_index][0]
    approx += u * table[origin_index][1]
    approx += (u * (u - 1) / 2) * table[origin_index - 1][2]
    approx += (u * (u - 1) * (u - 2) / 6) * table[origin_index - 1][3]

    return approx, table


x_vals = [25, 30, 35, 40]
y_vals = [0.2707, 0.3027, 0.3386, 0.3794]

value, fwd_table = gauss_forward_interp(x_vals, y_vals, 32)

print("Difference Table:")
for i in range(len(x_vals)):
    row = [f"{x_vals[i]:<3}"]
    for j in range(len(x_vals) - i):
        row.append(f"{fwd_table[i][j]:.4f}")
    print("\t".join(row))

print(f"\nInterpolated f(32) ≈ {value:.5f}")
