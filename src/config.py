import math
import numpy as np

INTERSECTION_RADIUS = 30  # m

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
FLOW_RATE = 0.5  # Vehicles per second
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

USE_LLM = True

# Possible locations that a vehicle is going to
POSSIBLE_LOCATIONS = [
    "Hospital",
    "Airport",
    "Home",
]

# Possible tasks that a vehicle is going to do, ranked by priority
# This may change priority if it's an important location but not urgent task
# I.e, going to a hospital but not in an emergency
POSSIBLE_TASKS = {
    "Hospital": [
        "go to the ER in an ambulance",
        "get stitches after a deep cut",
        "get a flu vaccine before winter",
    ],
    "Airport": [
        "catch a flight leaving in one hour",
        "catch a flight leaving in three hours",
        "pick up a relative",
    ],
    "Home": ["fix a burst pipe", "do laundry", "decorate for the holidays"],
}

GROUND_TRUTH_MATRIX = {
    "Hospital": {
        "go to the ER in an ambulance": 1,
        "get stitches after a deep cut": 4,
        "get a flu vaccine before winter": 7,
    },
    "Airport": {
        "catch a flight leaving in one hour": 2,
        "catch a flight leaving in three hours": 5,
        "pick up a relative": 6,
    },
    "Home": {
        "fix a burst pipe": 3,
        "do laundry": 8,
        "decorate for the holidays": 9,
    },
}

GROUND_TRUTH_FLATTEN = [
    task for location in GROUND_TRUTH_MATRIX for task in GROUND_TRUTH_MATRIX[location]
]
