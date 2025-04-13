# LLM Reasoning Policy
# The LLM should take the intention that was communciated between agents in an intersection zone
# All other agents will yield to a leader vehicle

# LLM should consider: Intention, waiting time, (possibly distance and velocity)
# I.e, an emergency vehicle should have the highest priority

import openai
import os
from config import Role, POSSIBLE_LOCATIONS
from BaseAuction import AuctionPolicy
import numpy as np
import random


class AgentLLM:
    def __init__(self, location=None):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if location is None:
            location = random.choice(POSSIBLE_LOCATIONS)
        content = f"""
You are going to {location}.
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

        if persist:
            self.messages.append(new_msg)
            self.messages.append(completion.choices[0].message)

        return completion.choices[0].message.content

    def add_other_agent(self, agent):
        prompt = f"""
There is another agent in the intersection zone now.
The other agent is going to {agent.task}.
"""

        code = self.query("system", prompt, persist=True)

    def get_priority(self, agent):
        prompt = f"""
You have communicated with the rest of the other agents in the intersection zone.
As a reminder, you are going to {agent.task}. If you have already communicated this, repeat it given any new vehicles.
You must determine your priority now. Output your priority as a number.
"""

        priority = self.query("system", prompt, persist=True)

        return priority


class LLMPolicy:
    def get_leader_followers(self, agents):
        # Sort the agents by priority
        # As a tiebreaker, use base priority
        priority_queue = sorted(
            agents,
            key=lambda x: (x.llm.get_priority(), AuctionPolicy.get_priority(x)),
            reverse=True,
        )

        priority_queue[0].role = Role.LEADER
        for agent in priority_queue[1:]:
            agent.role = Role.FOLLOWER
        return priority_queue[0], priority_queue[1:]
