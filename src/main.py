import gymnasium as gym
import highway_env
from MultiAgentWrapper import MultiAgentWrapper
import imageio
import random
import numpy as np
import argparse
import logging
import time

from util.CollisionAvoidance import inIntersectionZone, pastOrigin
from config import *
from policies.Policy import Policy
from util.Agent import Agent
from util.ActionTypeConversion import DiscreteToContinuous
from util.TurningDynamics import decelerate_follower, get_steering_angle

logger = logging.getLogger(__name__)
lg = logging.getLogger("V2V")
avg_v2v = []


def getArgs():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    ap.add_argument("-t", "--test", action="store_true", help="Run tests")
    ap.add_argument(
        "--max-iter", type=int, default=100, help="Max number of iterations"
    )
    ap.add_argument("-l", "--use_llm", action="store_true", help="Use LLM Policy")
    return ap.parse_args()


def V2V_communication(agentsInInt):
    start = time.time()
    for agent1 in agentsInInt:
        for agent2 in agentsInInt:
            # Sanity check, should not hit this
            if not (agent1.llm or agent2.llm):
                raise ValueError(f"One of the agents does not have its LLM initialized")
            if agent1 != agent2:
                # Let perpendicular agents know about each other
                dir_dot = np.dot(agent1.direction, agent2.direction)
                if not np.isclose(dir_dot, -1):
                    agent1.llm.add_other_agent(agent2)
                    agent2.llm.add_other_agent(agent1)
                elif not np.isclose(dir_dot, 1):
                    # Agent 1 is in front of agent 2
                    if np.linalg.norm(
                        agent1.position - np.array([0, 0])
                    ) < np.linalg.norm(agent2.position - np.array([0, 0])):
                        agent1.llm.add_other_agent_behind(agent2)
                        agent2.llm.add_other_agent_ahead(agent1)
                    else:
                        agent2.llm.add_other_agent_behind(agent1)
                        agent1.llm.add_other_agent_ahead(agent2)
    diff = time.time() - start
    avg_v2v.append(diff)
    lg.info(f"Time to calculate V2V communication: {diff:.4f} seconds")
    lg.info(
        f"Average time to calculate V2V communication: {np.mean(avg_v2v):.4f} seconds"
    )


def main():
    args = getArgs()
    fname = f"logs/{'llm' if args.use_llm else 'auction'}_{FLOW_RATE_HOUR}_full.log"
    logging.basicConfig(filename=fname, filemode="w+", level=logging.INFO)

    logger.info("Building multi agent simulation environment")
    env = MultiAgentWrapper()
    obs, info = env.reset()

    policy = Policy(use_llm=args.use_llm)
    logger.info(f"Using policy {policy.policy.__class__.__name__}")

    env.unwrapped.road.vehicles.clear()
    env.unwrapped.controlled_vehicles.clear()

    agents = []
    spawnedVehicles = []
    agentsInInt = []

    numOfCollisions = 0
    llm_requests_made = 0
    agents_reached_goal = 0
    total_time_to_goal = 0

    frames = []

    if args.debug:
        newVehicle = env.spawn_new_vehicle(spawnedVehicles, speed=MAX_SPEED)
        newVehicle.position = np.array([2.0, 100.0])
        newVehicle.endPoint = np.array([2.0, -100.0])
        newVehicle.heading = -3.14 / 2
        newVehicle.laneNum = 1
        agents.append(newVehicle)
        spawnedVehicles.append(newVehicle)

        newVehicle2 = env.spawn_new_vehicle(spawnedVehicles, speed=MAX_SPEED)
        newVehicle2.position = np.array([100, -2.0])
        newVehicle2.endPoint = np.array([-100.0, -2.0])
        newVehicle2.heading = -math.pi
        newVehicle2.laneNum = 3
        agents.append(newVehicle2)
        spawnedVehicles.append(newVehicle2)

        newVehicle3 = env.spawn_new_vehicle(spawnedVehicles, speed=MAX_SPEED)
        newVehicle3.position = np.array([2.0, 120.0])
        newVehicle3.endPoint = np.array([2.0, -100.0])
        newVehicle3.heading = -3.14 / 2
        newVehicle3.laneNum = 1
        agents.append(newVehicle3)
        spawnedVehicles.append(newVehicle3)

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

        if args.use_llm and len(agentsInInt) > 1:
            V2V_communication(agentsInInt)

        # Begin resolution of agents entering intersection
        # Of the agents in the intersection, one agent will remain at a constant velocity
        # The other agents will yield to the agent that is not yielding
        if len(agentsInInt) > 1:
            leader, followers = policy.get_leader_followers(agentsInInt)
            logger.info(f"Agent {leader.__repr__()} will lead")

            # Leader should not slow down
            a = MAX_ACCELERATION if leader.speed < MAX_SPEED else 0
            delta = get_steering_angle(leader) if leader.is_turning() else 0
            actions[agents.index(leader)] = [a, delta]

            # Followers will slow down based on vehicle in front of them in the priority queue
            for idx, follower in enumerate(followers):
                if follower.speed > 0:
                    a = decelerate_follower(
                        leader if idx == 0 else followers[idx - 1], follower
                    )
                    delta = get_steering_angle(agent) if agent.is_turning() else 0
                    actions[agents.index(follower)] = [a, delta]

        elif len(agentsInInt) == 1:
            # Only one agent in the intersection zone
            agent = agentsInInt[0]
            a = MAX_ACCELERATION if agent.speed < MAX_SPEED else 0
            delta = get_steering_angle(agent) if agent.is_turning() else 0
            actions[agents.index(agent)] = [a, delta]

        for i, agent in enumerate(agents):
            if agent not in agentsInInt or pastOrigin(
                agent.position, agent.direction, 0
            ):
                if agent.speed < MAX_SPEED:
                    actions[i] = [MAX_ACCELERATION, 0]
                else:
                    actions[i] = [0, 0]

        actions = tuple(actions)
        assert (
            len(actions) == env.num_agents
        ), f"Length of actions {len(actions)} does not match number of agents {env.num_agents}"
        for i, agent in enumerate(agents):
            agent.action = {"steering": actions[i][1], "acceleration": actions[i][0]}
            agent.timeToGoal += 1
        if len(actions) > 0:
            env.step(actions)

        # Spawn a new vehicle in environment
        if random.random() < FLOW_RATE / CYCLE_FREQUENCY:
            newVehicle = env.spawn_new_vehicle(spawnedVehicles, speed=MAX_SPEED)
            if newVehicle is None:
                continue
            agents.append(newVehicle)
            spawnedVehicles.append(newVehicle)
            logger.info(
                f"Spawning new vehicle {newVehicle.__repr__()} at iteration {iteration}"
            )

        for agent in agents:
            if env.reached_destination(agent):
                logger.info(
                    f"Removing agent {agent.__repr__()} at iteration {iteration}"
                )
                agents_reached_goal += 1
                total_time_to_goal += agent.timeToGoal
                if agent.llm:
                    llm_requests_made += agent.llm.requests_made
                env.despawn_vehicle(agent)
                if agent in spawnedVehicles:
                    spawnedVehicles.remove(agent)

        # Remove agents that have reached their destination
        agents = [a for a in agents if not env.reached_destination(a)]
        # Remove agents that have passed the intersection
        agentsInInt = [
            a
            for a in agentsInInt
            if not pastOrigin(a.position, a.direction) and a in agents
        ]
        logger.info(
            f"There exists a total of {env.num_agents} agents in the environment"
        )
        logger.info(f"Iteration {iteration} completed")

        if args.debug:
            env.render_mode = "human"
            env.render()
        if args.test:
            env.render_mode = "rgb_array"
            frames.append(env.render())

        colIdx = []
        for agent in agents:
            if agent.crashed or agent.hit:
                numOfCollisions += 1
                logger.info(
                    f"Agent {agent.__repr__()} has collided at iteration {iteration}"
                )
                if agent.llm:
                    llm_requests_made += agent.llm.requests_made
                env.despawn_vehicle(agent)
                if agent in spawnedVehicles:
                    spawnedVehicles.remove(agent)
                colIdx.append(agents.index(agent))
        # Remove agents that have crashed
        if len(colIdx) > 0:
            agents = [a for i, a in enumerate(agents) if i not in colIdx]
            agentsInInt = [
                a for i, a in enumerate(agentsInInt) if i not in colIdx and a in agents
            ]

    # Closing information
    iteration += 1
    logger.info(f"Logging closing information now")
    logger.info(f"Total number of iterations: {iteration}")
    logger.info(
        f"Simulated for {iteration / CYCLE_FREQUENCY} seconds at a rate of {FLOW_RATE} vehicles per second, {FLOW_RATE_HOUR} vehicles per hour"
    )
    logger.info(
        f"Average time to goal: {total_time_to_goal / agents_reached_goal:.4f} seconds"
    )
    logger.info(
        f"Throughput: {agents_reached_goal / (iteration / CYCLE_FREQUENCY):.4f} vehicles per minute"
    )
    if policy.use_llm:
        logger.info(
            f"Average time to calculate LLM priority: {policy.policy.avg_time:.4f} seconds"
        )
        logger.info(f"Number of calls to LLM: {policy.policy.num_calls}")
    logger.info(f"Total number of collisions: {numOfCollisions}")

    env.close()

    if args.test:
        imageio.mimsave(f"../res/test.mp4", frames, fps=30)


if __name__ == "__main__":
    main()
