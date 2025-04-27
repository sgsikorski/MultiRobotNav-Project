# First, install highway-env if you haven't already
# pip install highway-env

import gym
import highway_env
import numpy as np
from highway_env.envs.intersection_env import IntersectionEnv
from config import *
from TrafficLightEnv import TrafficLightIntersectionEnv


def main():
    env = TrafficLightIntersectionEnv()
    env.reset()

    for episode in range(1):
        obs = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()  # random actions
            obs, reward, terminated, truncated, info = env.step(action)
            # done = terminated or truncated
            env.render_mode = "human"
            env.render()

            print(
                f"Step: {env.steps}, Traffic Light: {info['traffic_light']}, Throughput: {info['throughput']:.2f}"
            )
            print(
                f"Step: {env.steps}, Traffic Light: {info['traffic_light']}, TTG: {info['ttg']:.2f}"
            )

    print(
        f"DONE\nStep: {env.steps}, Traffic Light: {info['traffic_light']}, Throughput: {info['throughput']:.2f}, TTG: {info['ttg']:.2f}"
    )
    env.close()


if __name__ == "__main__":
    main()
