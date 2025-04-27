# TODO: Implement the turning dynamics of the vehicle

import numpy as np
import math
from config import (
    MAX_ACCELERATION,
    MAX_STEERING,
    MIN_ACCELERATION,
    MIN_STEERING,
    TURN_RADIUS,
    LANE_OFFSET,
)


def decelerate_follower(leader, follower):
    dot_dir = np.dot(leader.direction, follower.direction)
    if np.isclose(dot_dir, -1) or np.isclose(dot_dir, 1):
        return 0
    follower_distance = np.linalg.norm(follower.position - np.array([0, 0]))
    leader_distance = np.linalg.norm(leader.position - np.array([0, 0]))
    t_leader = (leader_distance + leader.LENGTH + LANE_OFFSET) / leader.speed

    acceleration = np.min(
        (2 * (follower_distance - follower.speed * t_leader) / (t_leader**2))
    )
    acceleration = np.clip(acceleration, MIN_ACCELERATION, MAX_ACCELERATION)
    return acceleration


def get_steering_angle(agent):
    return math.atan2(agent.LENGTH, TURN_RADIUS)
    v_mag = math.hypot(agent.velocity[0], agent.velocity[1])
    if v_mag == 0:
        return 0.0

    dx = agent.endPoint[0] - agent.position[0]
    dy = agent.endPoint[1] - agent.position[1]
    if dx <= LANE_OFFSET * 2:
        dx = 0
    if dy <= LANE_OFFSET * 2:
        dy = 0
    distance_to_goal = np.linalg.norm([dx, dy])

    if distance_to_goal == 0:
        return 0.0

    dot = agent.direction[0] * dx + agent.direction[1] * dy
    det = agent.direction[0] * dy - agent.direction[1] * dx
    angle_to_goal = math.atan2(det, dot)

    delta = math.atan2(2 * agent.LENGTH * math.sin(angle_to_goal), distance_to_goal)
    return np.clip(delta, MIN_STEERING, MAX_STEERING)
