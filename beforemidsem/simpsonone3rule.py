import numpy as np

def f(x):
    return 1 / (1 + x**2)

a, b = 0, 10
n = 10   
h = (b - a) / n

x = np.linspace(a, b, n + 1)
y = f(x)

S = (h/3) * (y[0] + 4*sum(y[1:n:2]) + 2*sum(y[2:n-1:2]) + y[n])

print("Approximate Integral (Simpson's 1/3 Rule) =", S)

exact = np.arctan(10)
print("Exact Value =", exact)
print("Absolute Error =", abs(S - exact))
