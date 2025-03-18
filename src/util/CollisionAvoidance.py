import numpy as np


def inIntersectionZone(position, radius=25):
    origin = np.array([0, 0])
    return np.linalg.norm(position - origin) < radius


def pastOrigin(position, direction, distance=5):
    displacement = position - np.array([0, 0])
    projection = np.dot(displacement, direction)
    return projection > distance
