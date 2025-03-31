import math

# Vehicle Dynamics
MAX_SPEED = 10
MIN_SPEED = -10

MAX_ACCELERATION = 5  # m/s^2
MAX_STEERING = math.pi / 4.0  # rads
MIN_ACCELERATION = -5  # m/s^2
MIN_STEERING = -math.pi / 4.0  # rads
LANE_OFFSET = 2  # m

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
