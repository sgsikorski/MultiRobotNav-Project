# LLM Reasoning Policy
# The LLM should take the intention that was communciated between agents in an intersection zone
# All other agents will yield to a leader vehicle

# LLM should consider: Intention, waiting time, (possibly distance and velocity)
# I.e, an emergency vehicle should have the highest priority

import openai
import os
from config import Role, POSSIBLE_LOCATIONS
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
    def __init__(self, id, location=None):
        self.requests_made = 0
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if location is None:
            location = random.choice(POSSIBLE_LOCATIONS)
        # logger.info(f"LLM agent intialized going to {location}")
        content = f"""
You are going to {location}. You have the unique id {id}.
There are other agents going to a separate location. You will have communicate with each other.
You are all in an intersection zone and need to determine who has the highest priority.
A higher priority means you are going to go first.
If you are doing the same task or have the same priority then say so. 
Do not include pleasantries and be concise. 
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
They have the id {agent.id} and are going to {agent.task}.
"""

        code = self.query("user", prompt, persist=True)

    def add_other_agent_behind(self, agent):
        prompt = f"""
There is another agent in the intersection zone now and they are behind you.
They have the id {agent.id}. They will receive a lower priority than you. No matter what!
        """

        code = self.query("user", prompt, persist=True)

    def get_priority(self, agent):
        prompt = f"""
You have communicated with the rest of the other agents in the intersection zone.
As a reminder, you are going to {agent.task} and your id is {agent.id}. If you have already communicated this, repeat it given any new vehicles.
You must determine your priority now. Output your priority as a number.
"""

        priority = self.query("system", prompt, persist=True)

        logger.info(f"Agent {agent} priority: {priority}")
        return priority


class LLMPolicy:
    def __init__(self):
        self.avg_time = 0
        self.num_calls = 0

    def get_leader_followers(self, agents):
        # Sort the agents by priority
        # As a tiebreaker, use base priority
        start = time.time()
        priority_queue = sorted(
            agents,
            key=lambda x: (x.llm.get_priority(), AuctionPolicy.get_priority(x)),
            reverse=True,
        )
        diff = time.time() - start
        self.num_calls += 1
        self.avg_time = (self.avg_time * (self.num_calls - 1) + diff) / self.num_calls
        logger.info(f"Time to calculate LLM priority: {diff:.4f} seconds")

        priority_queue[0].role = Role.LEADER
        for agent in priority_queue[1:]:
            agent.role = Role.FOLLOWER
        return priority_queue[0], priority_queue[1:]
