import numpy as np

def pol2cart(rho, phi, origin=[0, 0]):
    r = -rho * np.sin(phi)
    c = rho * np.cos(phi)
    r += origin[0]
    c += origin[1]
    return (r, c)

def nonlinearity(x):
    return np.clip(abs(x) ** (1 / 3), 0.0, 1.0)
