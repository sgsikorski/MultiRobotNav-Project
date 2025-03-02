import gymnasium as gym
import highway_env
import numpy as np


class MultiAgentWrapper(gym.Env):

    def __init__(self, num_agents, env_name="intersection-v1"):
        super().__init__()
        self.num_agents = num_agents
        self.env = gym.make(env_name, render_mode="rgb_array")

        self.env.unwrapped.config.update(
            {
                "screen_width": 480,
                "screen_height": 640,
                "controlled_vehicles": self.num_agents,
                "centering_position": [0.5, 0.5],
                "scaling": 3,
                "show_trajectories": True,
                "spawn_probability": 0,
                "initial_vehicle_count": 0,
                "simulation_frequency": 5,
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
