# TODO: Implement the turning dynamics of the vehicle

import numpy as np
from config import MAX_ACCELERATION, MAX_STEERING, MIN_ACCELERATION, MIN_STEERING


def decelerate_follower(leader, follower):
    dot_dir = np.dot(leader.direction, follower.direction)
    if np.isclose(dot_dir, -1) or np.isclose(dot_dir, 1):
        return 0
    follower_distance = np.linalg.norm(follower.position - np.array([0, 0]))
    leader_distance = np.linalg.norm(leader.position - np.array([0, 0]))
    t_leader = (leader_distance + leader.LENGTH) / leader.velocity

    acceleration = np.min(
        (2 * (follower_distance - follower.velocity * t_leader) / (t_leader**2))
    )
    acceleration = np.clip(acceleration, MIN_ACCELERATION, MAX_ACCELERATION)
    return acceleration


# Given a vehicles position and velocity, return the acceleration and steering angle to turn left
def turnLeft(position, velocity):
    # Calculate the angle of the velocity vector
    angle = np.arctan2(velocity[1], velocity[0])
    # Calculate the angle of the vector from the vehicle to the origin
    angleToOrigin = np.arctan2(position[1], position[0])
    # Calculate the angle to turn
    angleToTurn = angleToOrigin - angle
    # Calculate the acceleration and steering angle
    acceleration = 1
    steering = 1

    acceleration = np.clip(acceleration, MIN_ACCELERATION, MAX_ACCELERATION)
    steering = np.clip(steering, MIN_STEERING, MAX_STEERING)
    return np.array([acceleration, steering])
