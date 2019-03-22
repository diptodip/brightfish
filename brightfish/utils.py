import numpy as np

def cart2pol(r, c, origin=[0, 0]):
    r -= origin[0]
    c -= origin[1]
    rho = np.sqrt(r**2 + c**2)
    phi = np.arctan2(c, r)
    return (rho, phi)

def pol2cart(rho, phi, origin=[0, 0]):
    r = rho * np.cos(phi)
    c = rho * np.sin(phi)
    r += origin[0]
    c += origin[1]
    return (r, c)
