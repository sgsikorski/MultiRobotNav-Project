from .CollisionAvoidance import pastOrigin
from highway_env.vehicle.controller import ControlledVehicle
import numpy as np
from ..policies.LLMPolicy import AgentLLM
from config import USE_LLM


class Agent(ControlledVehicle):
    def __init__(self, road, position, desination=None, **kwargs):
        super().__init__(road=road, position=position, **kwargs)
        self.pastIntersection = pastOrigin(self.position, self.direction)
        self.timeInZone = 0
        self.role = None
        # TODO: Change for possible turning
        self.endPoint = np.array([-1 * p if p != 2 else p for p in position])
        self.task = None
        self.id = hash(self)
        self.llm = AgentLLM(desination) if USE_LLM else None

    def __repr__(self):
        return f"Agent({self.position}, {self.speed})"
