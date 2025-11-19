s = [0, 10, 20, 30, 40, 50, 60]
v = [47, 58, 64, 65, 61, 52, 38]



inv_v = [1 / vi for vi in v]

# Apply trapezoidal rule
def trapezoidal_rule(x, y):
    n = len(x)
    integral = 0.0
    for i in range(1, n):
        integral += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2
    return integral

result = trapezoidal_rule(s, inv_v)
print("Integral approximation:",result)