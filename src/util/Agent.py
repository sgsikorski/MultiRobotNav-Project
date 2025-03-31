from .CollisionAvoidance import pastOrigin
from highway_env.vehicle.controller import ControlledVehicle
import numpy as np


class Agent(ControlledVehicle):
    def __init__(self, road, position, **kwargs):
        super().__init__(road=road, position=position, **kwargs)
        self.pastIntersection = pastOrigin(self.position, self.direction)
        self.timeInZone = 0
        self.role = None
        self.endPoint = np.array([-1 * p if p != 2 else p for p in position])

    def __repr__(self):
        return f"Agent({self.position}, {self.speed})"
