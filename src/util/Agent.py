from .CollisionAvoidance import pastOrigin
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
import numpy as np
from policies.LLMPolicy import AgentLLM
from config import USE_LLM, POSSIBLE_LOCATIONS, POSSIBLE_TASKS, ENDS
import random


class Agent(Vehicle):
    timeInZone = 0
    role = None
    goal_destination = None
    task = None
    llm = None

    def __init__(
        self,
        road,
        position,
        heading: float = 0,
        speed: float = 0,
        predition_type: str = "constant_steering",
    ):
        super().__init__(road, position, heading, speed, predition_type)
        self.pastIntersection = pastOrigin(self.position, self.direction)
        self.id = id(self)

    def __repr__(self):
        return f"Agent({self.position}, {self.speed})"

    def is_turning(self):
        to_goal = self.endPoint - self.position
        to_goal_normalized = to_goal / np.linalg.norm(to_goal)

        dot = np.dot(self.direction, to_goal_normalized)
        angle_diff = np.arccos(np.clip(dot, -1.0, 1.0))
        res = np.isclose(angle_diff, 0, atol=1e-2)
        return not np.isclose(angle_diff, 0, atol=1e-2)

    @staticmethod
    # Change this for an experimentation for LLM reasoning on location and task
    def decide_agent_task():
        destination = random.choice(POSSIBLE_LOCATIONS)
        task = random.choice(POSSIBLE_TASKS[destination])
        return destination, task
