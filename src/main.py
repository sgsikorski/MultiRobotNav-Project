import gymnasium as gym
import highway_env
from MultiAgentWrapper import MultiAgentWrapper
import imageio
import random
import numpy as np
import argparse
import logging

from util.CollisionAvoidance import inIntersectionZone, pastOrigin
from config import *
from policies.Policy import Policy
from util.Agent import Agent
from util.ActionTypeConversion import DiscreteToContinuous

logger = logging.getLogger(__name__)


def getArgs():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    ap.add_argument("-t", "--test", action="store_true", help="Run tests")
    ap.add_argument(
        "--max-iter", type=int, default=100, help="Max number of iterations"
    )
    ap.add_argument("-l", "--use_llm", action="store_true", help="Use LLM Policy")
    return ap.parse_args()


def main():
    logging.basicConfig(filename="logs/main.log", level=logging.INFO)
    args = getArgs()

    logger.info("Building multi agent simulation environment")
    env = MultiAgentWrapper(num_agents=0)
    obs, info = env.reset()

    policy = Policy(use_llm=args.use_llm)
    logger.info(f"Using policy {policy.policy.__class__.__name__}")

    # Clear any non controlled vehicles
    env.env.unwrapped.road.vehicles = []
    agents = []
    spawnedVehicles = []

    # newVehicle = env.spawn_new_vehicle(spawnedVehicles, speed=MAX_SPEED)
    # agents.append(Agent(newVehicle))
    # spawnedVehicles.append(newVehicle)
    # env.env.unwrapped.road.vehicles += spawnedVehicles

    frames = []

    agentsInInt = []
    logger.info(f"Starting experiment for {args.max_iter} iterations")
    for iteration in range(args.max_iter):
        actions = [[0, 0] for _ in range(env.num_agents)]
        for agent in agents:
            # Check which vehicles are approaching the intersection zone
            if inIntersectionZone(agent.position) and not pastOrigin(
                agent.position, agent.direction
            ):
                if agent not in agentsInInt:
                    agentsInInt.append(agent)
                else:
                    agentsInInt[agentsInInt.index(agent)].timeInZone += 1

        # TODO: Have the agents' LLMs communicate with each other

        # Begin resolution of agents entering intersection
        # Of the agents in the intersection, one agent will remain at a constant velocity
        # The other agents will yield to the agent that is not yielding
        if len(agentsInInt) > 1:
            leader, followers = policy.get_leader_followers(agentsInInt)
            logger.info(f"Agent {leader} will lead")
            lIdx = agentsInInt.index(leader)
            for idx in range(len(agentsInInt)):
                if idx != lIdx:
                    # Don't go backwards
                    if np.any(agentsInInt[idx].velocity > 0):
                        logger.info(f"Agent {agentsInInt[idx]} is slowing down")
                        actions[idx] = [-5, 0]
                else:
                    actions[idx] = [5, 0]
        # else:
        #     # Speed up any slowed down agents
        #     for agent in agents:
        #         if np.all(agent.velocity < MAX_SPEED):
        #             actions[agents.index(agent)] = [1, 0]

        actions = tuple(actions)
        assert (
            len(actions) == env.num_agents
        ), f"Length of actions {len(actions)} does not match number of agents {env.num_agents}"
        _obs, rewards, done, truncated, info = env.step(actions)

        # Spawn a new vehicle in environment
        if random.random() < FLOW_RATE / CYCLE_FREQUENCY:
            newVehicle = env.spawn_new_vehicle(spawnedVehicles, speed=MAX_SPEED)
            agents.append(newVehicle)
            spawnedVehicles.append(newVehicle)
            logger.info(f"Spawning new vehicle {newVehicle} at iteration {iteration}")

        for agent in agents:
            if env.reached_destination(agent):
                logger.info(f"Removing agent {agent} at iteration {iteration}")
                env.despawn_vehicle(agent)
                if agent in spawnedVehicles:
                    spawnedVehicles.remove(agent)

        # Remove agents that have reached their destination
        agents = [a for a in agents if not env.reached_destination(a)]
        # Remove agents that have passed the intersection
        agentsInInt = [
            a for a in agentsInInt if not pastOrigin(a.position, a.direction)
        ]
        env.env.unwrapped.road.vehicles += agents
        logger.info(
            f"There exists a total of {env.num_agents} agents in the environment"
        )
        logger.info(f"Iteration {iteration} completed")

        frames.append(env.render())
        obs = _obs

    env.close()

    if args.test:
        imageio.mimsave(f"../res/out.mp4", frames, fps=10)


if __name__ == "__main__":
    main()
