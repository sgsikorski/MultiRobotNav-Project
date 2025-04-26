import math
import numpy as np

# Vehicle Dynamics
MAX_SPEED = 10
MIN_SPEED = -10

MAX_ACCELERATION = 5  # m/s^2
MAX_STEERING = math.pi / 3.0  # rads
MIN_ACCELERATION = -5  # m/s^2
MIN_STEERING = -math.pi / 3.0  # rads
LANE_OFFSET = 2  # m
TURN_RADIUS = 10  # m

from enum import Enum, auto


class Role(Enum):
    LEADER = auto()
    FOLLOWER = auto()


class Action(Enum):
    NOP = auto()
    SLOW = auto()
    FAST = auto()
    LEFT = auto()
    RIGHT = auto()


CYCLE_FREQUENCY = 5  # Hz
FLOW_RATE = 1  # Vehicles per second
FLOW_RATE_HOUR = FLOW_RATE * 60 * 60  # Vehicles per hour

POSSIBLE_STARTS = [
    tuple([-2, -100]),
    tuple([2, 100]),
    tuple([-100, 2]),
    tuple([100, -2]),
]

STARTS = [
    np.array([-2, -100]),
    np.array([2, 100]),
    np.array([-100, 2]),
    np.array([100, -2]),
]

HEADINGS = [math.pi / 2, -math.pi / 2, 0, -math.pi]

ENDS = [
    np.array([-2, 100]),
    np.array([2, -100]),
    np.array([100, 2]),
    np.array([-100, -2]),
]

USE_LLM = False

# Possible locations that a vehicle is going to
POSSIBLE_LOCATIONS = [
    "Hospital",
    "Grocery Store",
    "School",
    "Work",
    "Airport",
    "Home",
]

# Possible tasks that a vehicle is going to do, ranked by priority
# This may change priority if it's an important location but not urgent task
# I.e, going to a hospital but not in an emergency
POSSIBLE_TASKS = {
    "Hospital": [
        "get stitches after a deep cut",
        "get an annual physical checkup",
        "get a flu vaccine before winter",
    ],
    "Grocery Store": [
        "buy medicine for a fever",
        "do weekly grocery shopping",
        "get snacks for movie night",
    ],
    "School": [
        "take a final exam",
        "submit a project",
        "attend a voluntary club meeting",
    ],
    "Work": ["present at a team meeting", "pick up a work laptop", "clean out desk"],
    "Airport": [
        "travel for a job interview",
        "catch a flight for a wedding",
        "pick up a relative",
    ],
    "Home": ["fix a burst pipe", "do laundry", "decorate for the holidays"],
}
