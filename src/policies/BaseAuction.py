# Basic Game Theory Auctioning Policy
# This should be the bidding based on distance and velocity to the intersection
# This is used as a fall back strategy when the LLM policy is not able to make a decision
import numpy as np
from config import Role


class AuctionPolicy:

    # Return the leader agent and a list of the followers
    def get_leader_followers(self, agents) -> tuple[dict, list]:
        priorities = []
        for agent in agents:
            priorities += [self.get_priority(agent)]

        # Sort the agents by priority
        priorityIdx = np.argsort(priorities)
        priority_queue = [agents[idx] for idx in priorityIdx]

        priority_queue[0].role = Role.LEADER
        for agent in priority_queue[1:]:
            agent.role = Role.FOLLOWER
        return priority_queue[0], priority_queue[1:]

    @staticmethod
    def get_priority(agent) -> float:
        # p = (v + c) / s * w
        distanceToOrigin = np.linalg.norm(agent.position - np.array([0, 0]))
        speed = np.linalg.norm(agent.velocity)
        priority = (speed + 1 / (distanceToOrigin + 1e-6)) * agent.timeInZone
        return priority
