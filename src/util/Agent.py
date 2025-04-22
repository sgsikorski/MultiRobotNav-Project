from .CollisionAvoidance import pastOrigin
from highway_env.vehicle.controller import ControlledVehicle
import numpy as np
from policies.LLMPolicy import AgentLLM
from config import USE_LLM
import random


class Agent(ControlledVehicle):
    def __init__(self, road, position, desination=None, **kwargs):
        super().__init__(road=road, position=position, **kwargs)
        self.pastIntersection = pastOrigin(self.position, self.direction)
        self.timeInZone = 0
        self.role = None
        self.endPoint = np.array([self.get_endpoint(p) for p in position])
        self.task = None
        self.id = hash(self)
        self.llm = AgentLLM(self.id, desination) if USE_LLM else None

    def __repr__(self):
        return f"Agent({self.position}, {self.speed})"

    def get_endpoint(self, position):
        # TODO: Implement turning dynamics first
        # # Decide that the vehicle is turning
        # if random.random() < 0.2:
        #     position[1], position[0] = position[0], position[1]
        return -1 * position if int(position) != 2 else position
