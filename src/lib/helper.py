import numpy as np

def derivative1st(y, x):
    h = np.diff(x)
    if y.ndim == 2:
        h = h.reshape(-1, 1)
    elif y.ndim == 3:
        h = h.reshape(-1, 1, 1)
    dy = np.empty_like(y)
    a = np.divide(h[1:] - h[:-1], h[:-1] * h[1:])
    b = -np.divide(h[1:], h[:-1] * (h[:-1] + h[1:]))
    c = np.divide(h[:-1], h[1:] * (h[:-1] + h[1:]))
    dy[1:-1] = a * y[1:-1] + b * y[:-2] + c * y[2:]
    a = -(2 * h[0] + h[1]) / (h[0] * (h[0] + h[1]))
    b = (h[0] + h[1]) / (h[0] * h[1])
    c = -h[0] / (h[1] * (h[0] + h[1]))
    dy[0] = a * y[0] + b * y[1] + c * y[2]
    a = h[-1] / (h[-2] * (h[-1] + h[-2]))
    b = -(h[-2] + h[-1]) / (h[-2] * h[-1])
    c = (h[-2] + 2. * h[-1]) / (h[-1] * (h[-1] + h[-2]))
    dy[-1] = a * y[-3] + b * y[-2] + c * y[-1]
    return dy 


def derivative2nd(y, x):
    h = np.diff(x)
    if y.ndim == 2:
        h = h.reshape(-1, 1)
    elif y.ndim == 3:
        h = h.reshape(-1, 1, 1)
    ddy = np.zeros_like(y)
    a = -np.divide(2., h[:-1] * h[1:])
    b = np.divide(2., h[1:] * (h[:-1] + h[1:]))
    c = np.divide(2., h[:-1] * (h[:-1] + h[1:]))
    ddy[1:-1] = a * y[1:-1] + b * y[:-2] + c * y[2:]
    '''
    a = 2. * (3. * h[0] + 2. * h[1] + h[2]) / (h[0] * (h[0] + h[1]) * (h[0] + h[1] + h[2]))
    b = -2. * (2. * h[0] + 2. * h[1] + h[2]) / (h[0] * h[1] * (h[1] + h[2]))
    c = 2. * (2. * h[0] + h[1] +h[2]) / (h[1] * h[2] * (h[0] + h[1]))
    d = 2. * (2. * h[0] + h[1]) / (h[2] * (h[1] + h[2]) * (h[0] + h[1] + h[2]))
    ddy[0] = a * y[0] + b * y[1] + c * y[2] + d * y[3]
    a = 2. * (3. * h[-1] + 2. * h[-2] + h[-3]) / (h[-1] * (h[-1] + h[-2]) * (h[-1] + h[-2] + h[-3]))
    b = -2. * (2. * h[-1] + 2. * h[-2] + h[-3]) / (h[-1] * h[-2] * (h[-2] + h[-3]))
    c = 2. * (2. * h[-1] + h[-2] + h[-3]) / (h[-2] * h[-3] * (h[-1] + h[-2]))
    d = 2. * (2. * h[-1] + h[-2]) / (h[-3] * (h[-2] + h[-3]) * (h[-1] + h[-2] + h[-3]))
    ddy[-1] = a * y[-1] + b * y[-2] + c * y[-3] + d * y[-4]
    '''
    return ddy