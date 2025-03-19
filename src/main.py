import gymnasium as gym
import highway_env
from MultiAgentWrapper import MultiAgentWrapper
import imageio
import random
import numpy as np
import argparse

from util.CollisionAvoidance import inIntersectionZone, pastOrigin
from config import *
from policies.Policy import Policy
from util.Agent import Agent


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
    ap.add_argument("-l", "--use_llm", action="store_true", help="Use LLM Policy")
    return ap.parse_args()


def main():
    args = getArgs()

    env = MultiAgentWrapper(num_agents=0)
    obs, info = env.reset()

    policy = Policy(use_llm=args.use_llm)

    # Clear any non controlled vehicles
    env.env.unwrapped.road.vehicles = []
    agents = [Agent(a) for a in env.env.unwrapped.road.vehicles]
    spawnedVehicles = []

    newVehicle = env.spawn_new_vehicle(speed=MAX_SPEED)
    agents.append(Agent(newVehicle))
    spawnedVehicles.append(newVehicle)
    # env.env.unwrapped.road.vehicles += spawnedVehicles

    frames = []

    intersection_queue = []
    agentsInInt = []
    for iteration in range(args.max_iter):
        actions = [[0, 0] for _ in range(env.num_agents)]
        for agent in agents:
            # Check which vehicles are approaching the intersection zone
            if inIntersectionZone(agent.position) and not pastOrigin(
                agent.position, agent.direction
            ):
                if args.debug:
                    print(f"Agent {agent} is in the intersection zone")
                if agent not in agentsInInt:
                    agentsInInt.append(Agent(agent))
                else:
                    agentsInInt[agentsInInt.index(agent)].timeInZone += 1

        # Begin resolution of agents entering intersection
        # Of the agents in the intersection, one agent will remain at a constant velocity
        # The other agents will yield to the agent that is not yielding
        if len(agentsInInt) > 1:
            # TODO: Determine lane path and destination, i.e parallel agents don't need to yield
            leader, followers = policy.get_leader_followers(agentsInInt)
            if args.debug:
                print(f"Agent {leader.id} will lead")
            lIdx = agents.index(leader.agent)
            for idx in range(len(agentsInInt)):
                if idx != lIdx:
                    # Don't go backwards
                    if np.any(agentsInInt[idx].velocity > 0):
                        actions[idx] = [-1, 0]
            intersection_queue = followers
        else:
            # Speed up previous yielding agents
            if len(intersection_queue) > 0:
                yieldedAgent = intersection_queue[0]
                if np.any(yieldedAgent.velocity < MAX_SPEED):
                    actions[agents.index(yieldedAgent.agent)] = [1, 0]

        actions = tuple(actions)
        assert len(actions) == env.num_agents
        _obs, rewards, done, truncated, info = env.step(actions)

        # Spawn a new vehicle in environment
        # TODO: Adjust probability and sim config for testing
        if random.random() < 0.3:
            newVehicle = env.spawn_new_vehicle(speed=MAX_SPEED)
            agents.append(Agent(newVehicle))
            spawnedVehicles.append(newVehicle)
        env.env.unwrapped.road.vehicles += spawnedVehicles

        frames.append(env.render())
        obs = _obs

        # Remove agents that have reached their destination
        agents = [
            agent
            for agent in agents
            if np.all(agent.position == agent.agent.destination)
        ]
        # Remove agents that have passed the intersection
        agentsInInt = [
            a for a in agentsInInt if not pastOrigin(a.position, a.direction)
        ]

    env.close()

    if args.test:
        imageio.mimsave(f"../res/out.mp4", frames, fps=2)


if __name__ == "__main__":
    main()
