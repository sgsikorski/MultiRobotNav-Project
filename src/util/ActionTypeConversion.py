from config import *
import numpy as np

ACTIONS = ["NOP", "SLOW", "FAST", "LEFT", "RIGHT"]


def DiscreteToContinuous(action: Action):
    if action == Action.FAST:
        return np.array([MAX_ACCELERATION, 0])
    elif action == Action.SLOW:
        return np.array([MIN_ACCELERATION, 0])
    elif action == Action.RIGHT:
        return np.array([0, MAX_STEERING])
    elif action == Action.LEFT:
        return np.array([0, MIN_STEERING])
    else:
        return np.array([0, 0])


# Just in case if needed
def ContinuousToDiscrete(action):
    return
