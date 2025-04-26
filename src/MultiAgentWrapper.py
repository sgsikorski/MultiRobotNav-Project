import gymnasium as gym
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.envs import IntersectionEnv
from util.Agent import Agent
import numpy as np
from config import *


class MultiAgentWrapper(IntersectionEnv):
    num_agents = 0

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(
            {
                "screen_width": 480,
                "screen_height": 640,
                "centering_position": [0.5, 0.5],
                "scaling": 3,
                "show_trajectories": True,
                "spawn_probability": 0,
                "initial_vehicle_count": 0,
                "simulation_frequency": CYCLE_FREQUENCY,
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {
                        "type": "Kinematics",
                    },
                    "observe_intentions": True,
                },
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "ContinuousAction",
                        "steering_range": [MIN_STEERING, MAX_STEERING],
                        "longitudinal": True,
                        "lateral": True,
                        "dynamical": True,
                    },
                },
                "vehicles": {
                    "speed_range": [MIN_SPEED, MAX_SPEED],
                },
            }
        )
        return config

    def render(self):
        viewer = self.unwrapped.viewer
        if viewer:
            viewer.window_position = self.window_position.__get__(viewer, type(viewer))
        return super().render()

    # Function overload to always center on the origin of the screen
    # This is meant for the intersection environment
    def window_position(self):
        return np.array([0, 0])

    # Spawn a new vehicle at the start of one of the four lanes
    def spawn_new_vehicle(self, spawnedVehicles, speed=10):
        possibleStarts = list(
            set(POSSIBLE_STARTS) - {tuple(agent.position) for agent in spawnedVehicles}
        )
        if len(possibleStarts) == 0:
            return None
        position = possibleStarts[np.random.randint(0, len(possibleStarts))]
        posIdx = POSSIBLE_STARTS.index(position)

        vehicle = Agent(
            self.unwrapped.road,
            np.array(position),
            speed=speed,
        )
        vehicle.position = position
        vehicle.endPoint = ENDS[posIdx]
        vehicle.heading = HEADINGS[posIdx]
        self.num_agents += 1
        if vehicle not in self.unwrapped.road.vehicles:
            self.unwrapped.road.vehicles.append(vehicle)
        self.unwrapped.controlled_vehicles.append(vehicle)
        return vehicle

    def despawn_vehicle(self, agent):
        self.num_agents -= 1
        if agent in self.unwrapped.road.vehicles:
            self.unwrapped.road.vehicles.remove(agent)
        if agent in self.unwrapped.controlled_vehicles:
            self.unwrapped.controlled_vehicles.remove(agent)
        return True

    def reached_destination(self, agent):
        return np.all(np.abs(agent.position - agent.endPoint) < 5)
