from .CollisionAvoidance import pastOrigin
from highway_env.vehicle.controller import ControlledVehicle
import numpy as np
from policies.LLMPolicy import AgentLLM
from config import USE_LLM, POSSIBLE_LOCATIONS, POSSIBLE_TASKS
import random


class Agent(ControlledVehicle):
    def __init__(self, road, position, **kwargs):
        super().__init__(road=road, position=position, **kwargs)
        self.pastIntersection = pastOrigin(self.position, self.direction)
        self.timeInZone = 0
        self.role = None
        self.endPoint = np.array([self.get_endpoint(p) for p in position])
        self.id = hash(self)
        self.goal_destination, self.task = self.decide_agent_task()
        self.llm = (
            AgentLLM(self.id, self.goal_destination, self.task) if USE_LLM else None
        )

    def __repr__(self):
        return f"Agent({self.position}, {self.speed})"

    def get_endpoint(self, position):
        # TODO: Implement turning dynamics first
        # # Decide that the vehicle is turning
        # if random.random() < 0.2:
        #     position[1], position[0] = position[0], position[1]
        return -1 * position if int(position) != 2 else position

    # Change this for an experimentation for LLM reasoning on location and task
    def decide_agent_task(self):
        destination = random.choice(POSSIBLE_LOCATIONS)
        task = random.choice(POSSIBLE_TASKS[destination])
        return destination, task
