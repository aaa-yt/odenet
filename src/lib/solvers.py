import numpy as np

def euler(func, t, x0, args=None):
    if args is not None:
        func = lambda t, x, func=func: func(t, x, *args)
    solution = np.empty(shape=(len(t), len(x0), len(x0[0])), dtype=np.float32)
    solution[0] = x0
    for i, dt in enumerate(np.diff(t)):
        solution[i+1] = solution[i] + dt * func(t[i], solution[i])
    return solution


def RK4(func, t, x0, args=None):
    if args is not None:
        func = lambda t, x, func=func: func(t, x, *args)
    solution = np.empty(shape=(len(t), len(x0), len(x0[0])), dtype=np.float32)
    solution[0] = x0
    x = x0
    for i, dt in enumerate(np.diff(t)):
        k1 = func(t[i], x)
        k2 = func(t[i] + dt * 0.5, x + dt * k1 * 0.5)
        k3 = func(t[i] + dt * 0.5, x + dt * k2 * 0.5)
        k4 = func(t[i] + dt, x + dt * k3)
        x = x + dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        solution[i+1] = x
    return solution

def get(solver_type):
    if solver_type.lower() == "rk4":
        return RK4
    else:
        return euler