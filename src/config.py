# Vehicle Dynamics
MAX_SPEED = 10
MIN_SPEED = -10

MAX_ACCELERATION = 1
MAX_STEERING = 1
MIN_ACCELERATION = -1
MIN_STEERING = -1
LANE_OFFSET = 2

from enum import Enum, auto


class Role(Enum):
    LEADER = auto()
    FOLLOWER = auto()
