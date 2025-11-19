def f(x):
    return x**3 - 3*x + 1

def bisection(a, b, tol=0.001):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Function has same signs at interval endpoints. Bisection not possible.")

    iteration = 0
    print(f"{'Iter':<5}{'a':<12}{'b':<12}{'mid':<12}{'f(mid)':<15}{'Interval Width'}")

    while (b - a) > tol:
        iteration += 1
        mid = (a + b) / 2
        fm = f(mid)
        print(f"{iteration:<5}{a:<12.6f}{b:<12.6f}{mid:<12.6f}{fm:<15.9f}{(b-a):.6f}")

        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm

    root = (a + b) / 2
    print("\nApproximate root (3 d.p.):", round(root, 3))
    return round(root, 3)

bisection(1, 2)
