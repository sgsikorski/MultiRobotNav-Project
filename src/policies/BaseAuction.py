# Basic Game Theory Auctioning Policy
# This should be the bidding based on distance and velocity to the intersection
# This is used as a fall back strategy when the LLM policy is not able to make a decision
import numpy as np
from config import Role, GROUND_TRUTH_MATRIX, GROUND_TRUTH_FLATTEN
import logging
from functools import cmp_to_key
from util.CollisionAvoidance import pastOrigin

logger = logging.getLogger(__name__)


class AuctionPolicy:

    # Return the leader agent and a list of the followers
    def get_leader_followers(self, agents) -> tuple[dict, list]:
        def compare(a, b):
            p_a = self.get_priority(a)
            p_b = self.get_priority(b)
            d_a = np.linalg.norm(a.position - np.array([0, 0]))
            d_b = np.linalg.norm(b.position - np.array([0, 0]))

            if p_a > p_b:
                if a.laneNum == b.laneNum:
                    return 1 if d_a > d_b else -1
                return -1
            elif p_a < p_b:
                if a.laneNum == b.laneNum:
                    return 1 if d_a > d_b else -1
                return 1
            else:
                return 0

        priority_queue = sorted(
            agents,
            key=cmp_to_key(compare),
        )

        # Quick comparsion
        determinedTaskPriority = []
        for agent in priority_queue:
            determinedTaskPriority.append(
                GROUND_TRUTH_MATRIX[agent.goal_destination][agent.task]
            )

        is_sorted = np.all(
            [
                determinedTaskPriority[i] <= determinedTaskPriority[i + 1]
                for i in range(len(determinedTaskPriority) - 1)
            ]
        )

        if not is_sorted:
            for i in range(len(determinedTaskPriority) - 1):
                current_rank = determinedTaskPriority[i] - 1
                next_rank = determinedTaskPriority[i + 1] - 1
                if current_rank > next_rank:
                    logger.warning(
                        f"  '{GROUND_TRUTH_FLATTEN[current_rank]}' (rank {current_rank}) should come after '{GROUND_TRUTH_FLATTEN[next_rank]}' (rank {next_rank})"
                    )
        else:
            logger.info("All tasks are in the correct order.")

        priority_queue[0].role = Role.LEADER
        for agent in priority_queue[1:]:
            agent.role = Role.FOLLOWER
        return priority_queue[0], priority_queue[1:]

    @staticmethod
    def get_priority(agent) -> float:
        # p = (v + c) / s * w
        if pastOrigin(agent.position, agent.direction, 0):
            return 100
        distanceToOrigin = np.linalg.norm(agent.position - np.array([0, 0]))
        speed = np.linalg.norm(agent.velocity)
        priority = (speed / distanceToOrigin) * (agent.timeInZone + 1)
        return priority
