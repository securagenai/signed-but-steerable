"""Geometry of the vector-text desynchronisation attack (E7)."""
import numpy as np


def rotate_towards(e, q, max_angle):
    """Unit vector at angle <= max_angle from the honest vector e that maximises the cosine with the
    target q: rotation of e towards q in the plane span(e, q). Optimal in closed form, so a device
    check that accepts cos(shipped, recomputed) >= tau gives the attacker exactly arccos(tau)."""
    c = float(np.clip(e @ q, -1, 1))
    phi = np.arccos(c)
    if phi < 1e-9:
        return e.copy()
    u = q - c * e
    u = u / np.linalg.norm(u)
    alpha = min(max_angle, phi)
    return np.cos(alpha) * e + np.sin(alpha) * u


def budget(tau):
    """Angular budget allowed by a tolerance tau on the cosine; tau < 0 means no check at all."""
    return np.pi if tau < 0 else float(np.arccos(np.clip(tau, -1, 1)))
