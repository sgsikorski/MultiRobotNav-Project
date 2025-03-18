import gymnasium as gym
import highway_env
from MultiAgentWrapper import MultiAgentWrapper
import imageio
import random
import numpy as np
import argparse

from util.CollisionAvoidance import inIntersectionZone, pastOrigin
from config import *


def getArgs():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    ap.add_argument("-t", "--test", action="store_true", help="Run tests")
    ap.add_argument(
        "-n",
        "--num-agents",
        type=int,
        default=2,
        help="Number of agents in the environment",
    )
    ap.add_argument(
        "--max-iter", type=int, default=100, help="Max number of iterations"
    )
    return ap.parse_args()


def main():
    args = getArgs()

    env = MultiAgentWrapper(num_agents=args.num_agents)
    obs, info = env.reset()

    # Clear any non controlled vehicles
    env.env.unwrapped.road.vehicles = []
    agents = env.env.unwrapped.road.vehicles
    spawnedVehicles = []

    newVehicle = env.spawn_new_vehicle(speed=MAX_SPEED)
    agents.append(newVehicle)
    spawnedVehicles.append(newVehicle)
    env.env.unwrapped.road.vehicles += spawnedVehicles

    frames = []

    intersection_queue = []
    for iteration in range(args.max_iter):
        agentsInInt = []
        actions = [[0, 0] for _ in range(env.num_agents)]
        for agent in agents:
            # Check which vehicles are approaching the intersection zone
            if inIntersectionZone(agent.position) and not pastOrigin(
                agent.position, agent.direction
            ):
                if args.debug:
                    print(f"Agent {agent} is in the intersection zone")
                agentsInInt.append(
                    {
                        "agent": agent,
                        "id": agent.__hash__(),
                        "position": agent.position,
                        "velocity": agent.velocity,
                        "pastIntersection": pastOrigin(agent.position, agent.direction),
                        "direction": agent.direction,
                    }
                )

        # Begin resolution of agents entering intersection
        # Of the agents in the intersection, one agent will remain at a constant velocity
        # The other agents will yield to the agent that is not yielding
        if len(agentsInInt) > 1:
            # TODO: Determine yielding agent from LLMPolicy or the BaseAuction Policy
            # TODO: Determine lane path and destination, i.e parallel agents don't need to yield
            agentToYield = agentsInInt[0]
            if args.debug:
                print(f"Agent {agentToYield['id']} will yield")
            actions[agents.index(agentToYield["agent"])] = [-1, 0]
            intersection_queue.append(agentToYield)
        else:
            # Speed up previous yielding agents
            if len(intersection_queue) > 0:
                yieldedAgent = intersection_queue[0]
                if np.any(yieldedAgent["velocity"] < MAX_SPEED):
                    actions[agents.index(yieldedAgent["agent"])] = [1, 0]

        actions = tuple(actions)
        assert len(actions) == env.num_agents
        _obs, rewards, done, truncated, info = env.step(actions)

        if random.random() < 0.5:
            newVehicle = env.spawn_new_vehicle(speed=MAX_SPEED)
            agents.append(newVehicle)
            spawnedVehicles.append(newVehicle)
        env.env.unwrapped.road.vehicles += spawnedVehicles

        frames.append(env.render())
        obs = _obs

    env.close()

    if args.test:
        imageio.mimsave(f"../res/out.mp4", frames, fps=2)


if __name__ == "__main__":
    main()
