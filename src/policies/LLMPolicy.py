# TODO: LLM Reasoning Policy
# The LLM should take the intention that was communciated between agents in an intersection zone
# All other agents will yield to a leader vehicle

# LLM should consider: Intention, waiting time, (possibly distance and velocity)
# I.e, an emergency vehicle should have the highest priority

import openai
import os
from config import Role

openai.api_key = os.getenv("OPENAI_API_KEY")


class LLMPolicy:
    def __init__(self, location):
        # TODO: Possibly modify this
        content = f"""
You are going to {location}.
There is another agent going to a location. 
You will have a conversation until you determine whether you have more, less, or the same priority as them depending on the task you and they are performing. 
If you are doing the same task or have the same priority then say so. 
Do not include pleasantries and be concise. 
Once you have reached a consensus with the other agent, output the number 1 and nothing else. 
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

    def get_role(self):
        prompt = """
You have come to a consensus. 
It is vital you remember what you agreed on with the other agent! 
If your task is more important than the other agent's task, output the number 2. 
If your task is less important than the other agent's task, output the number 3.
""".strip()

        code = self.query("system", prompt, persist=False)

        if "2" in code:
            return Role.LEADER
        if "3" in code:
            return Role.FOLLOWER
        if "4" in code:
            return None

    def get_leader_followers(self, agents):
        raise NotImplementedError(
            "get_leader_followers for the designated policy is not implemented yet"
        )
