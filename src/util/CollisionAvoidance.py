import numpy as np
from config import INTERSECTION_RADIUS


def inIntersectionZone(position, radius=INTERSECTION_RADIUS):
    origin = np.array([0, 0])
    return np.linalg.norm(position - origin) < radius


def pastOrigin(position, direction, distance=5):
    displacement = position - np.array([0, 0])
    projection = np.dot(displacement, direction)
    return projection > distance
