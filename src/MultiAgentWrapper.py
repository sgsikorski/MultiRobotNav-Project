import gymnasium as gym
from highway_env.vehicle.controller import ControlledVehicle
from util.Agent import Agent
import numpy as np
from config import *


class MultiAgentWrapper(gym.Env):

    def __init__(self, num_agents, env_name="intersection-v1"):
        super().__init__()
        self.num_agents = num_agents
        self.env = gym.make(env_name, render_mode="rgb_array")

        self.env.unwrapped.config.update(
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
                    },
                },
                "vehicles": {
                    "speed_range": [MIN_SPEED, MAX_SPEED],
                },
            }
        )

        self.observation_space = gym.spaces.Tuple(
            [self.env.observation_space] * self.num_agents
        )

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def render(self, mode="human"):
        viewer = self.env.unwrapped.viewer
        if viewer:
            viewer.window_position = self.window_position.__get__(viewer, type(viewer))
        return self.env.render()

    def close(self):
        self.env.close()

    # Function overload to always center on the origin of the screen
    # This is meant for the intersection environment
    def window_position(self):
        return np.array([0, 0])

    # Spawn a new vehicle at the start of one of the four lanes
    def spawn_new_vehicle(self, spawnedVehicles, speed=10):
        starts = {
            tuple([2, -100]),
            tuple([-100, 2]),
            tuple([2, 100]),
            tuple([100, 2]),
        }
        possibleStarts = list(
            starts - {tuple(agent.position) for agent in spawnedVehicles}
        )
        if len(possibleStarts) == 0:
            return None
        position = possibleStarts[np.random.randint(0, len(possibleStarts))]
        vehicle = Agent(
            self.env.unwrapped.road,
            np.array(position),
            speed=speed,
        )
        self.num_agents += 1
        return vehicle

    def despawn_vehicle(self, agent):
        self.num_agents -= 1
        return True

    def reached_destination(self, agent):
        return np.all(np.abs(agent.position - agent.endPoint) < 5)
