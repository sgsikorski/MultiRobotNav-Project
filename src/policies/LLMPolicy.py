# LLM Reasoning Policy
# The LLM should take the intention that was communciated between agents in an intersection zone
# All other agents will yield to a leader vehicle

# LLM should consider: Intention, waiting time, (possibly distance and velocity)
# I.e, an emergency vehicle should have the highest priority

import openai
import os
from config import (
    Role,
    POSSIBLE_LOCATIONS,
    LANE_OFFSET,
    POSSIBLE_TASKS,
    GROUND_TRUTH_MATRIX,
    GROUND_TRUTH_FLATTEN,
)
from policies.BaseAuction import AuctionPolicy
import numpy as np
import random
import logging
import time

logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. Please set it to use the LLM."
    )


class AgentLLM:
    def __init__(self, agent, location=None, task=None):
        self.requests_made = 0
        if location is None:
            location = random.choice(POSSIBLE_LOCATIONS)
        if task is None:
            task = random.choice(POSSIBLE_LOCATIONS[location])

        logger.info(f"LLM agent intialized going to {location}")
        content = f"""
You are going to {location} to {task}. You have the unique id {agent.id}.
There are other agents going to a separate location and have other tasks of various importance.
You will have communicate with each other.
You are all in an intersection zone and need to determine who has the highest priority.
A higher priority means you are going to go first. 1 is the highest priority.
If you are doing the same task or have the same priority then say so. 
Do not include pleasantries and be concise.
Consider your current dynamics:
Current speed: {agent.speed}
Current distance until you're not in the intersection: {np.linalg.norm(agent.position-np.array([0, 0])) + agent.LENGTH + LANE_OFFSET}
Time spent in the intersection zone: {agent.timeInZone}
Remembering your task correctly is paramount!
""".strip()
        initial_msg = {"role": "system", "content": content}
        self.messages = [initial_msg]

    def query(self, role, prompt, persist=True):
        new_msg = {"role": role, "content": prompt}

        completion = openai.chat.completions.create(
            model="gpt-4o-mini", messages=self.messages + [new_msg]
        )
        self.requests_made += 1

        if persist:
            self.messages.append(new_msg)
            self.messages.append(completion.choices[0].message)

        return completion.choices[0].message.content

    def add_other_agent(self, agent):
        prompt = f"""
There is another agent in the intersection zone now.
They have the id {agent.id} and are going to {agent.goal_destination} to {agent.task}.
Consider their dynamics:
Current speed: {agent.speed}
Current distance until they're not in the intersection: {np.linalg.norm(agent.position-np.array([0, 0])) + agent.LENGTH + LANE_OFFSET}
Time spent in the intersection zone: {agent.timeInZone}
You should focus on their desination and task and value those more when determing your priority!
"""

        code = self.query("user", prompt, persist=True)

    def add_other_agent_behind(self, agent):
        prompt = f"""
There is another agent in the intersection zone now and they are behind you.
They have the id {agent.id}. They will receive a lower priority than you. No matter what!
"""

        code = self.query("user", prompt, persist=True)

    def add_other_agent_ahead(self, agent):
        prompt = f"""
There is another agent in the intersection zone now and they are ahead you.
They have the id {agent.id}. They will receive a higher priority than you. No matter what!
"""

        code = self.query("user", prompt, persist=True)

    def get_priority(self, agent):
        prompt = f"""
You have communicated with the rest of the other agents in the intersection zone.
As a reminder, you are going to {agent.goal_destination} to {agent.task} and your id is {agent.id}.
If you have already communicated this, revaluate your priority given agents that no longer communciate and new agents that have.
You must determine your priority now. A higher priority means you are going to go first. 1 is the highest priority.
Output the priority as a number. Only output the number!
"""

        priority = self.query("user", prompt, persist=True)

        logger.info(f"Agent {agent.__repr__()} priority: {priority}")
        return priority


class LLMPolicy:
    def __init__(self):
        self.avg_time = 0
        self.num_calls = 0

    def get_leader_followers(self, agents):
        # Sort the agents by priority
        # As a tiebreaker, use base priority
        priorities = set()

        def key_sort(x):
            start = time.time()
            model_p = x.llm.get_priority(x)
            diff = time.time() - start
            self.num_calls += 1
            self.avg_time = (
                self.avg_time * (self.num_calls - 1) + diff
            ) / self.num_calls
            logger.info(f"Time to calculate LLM priority: {diff:.4f} seconds")
            auction_p = AuctionPolicy.get_priority(x)
            if model_p in priorities:
                logging.info(f"Fallback policy used")
            else:
                priorities.add(model_p)
            return (model_p, auction_p)

        priority_queue = sorted(
            agents,
            key=key_sort,
        )

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
