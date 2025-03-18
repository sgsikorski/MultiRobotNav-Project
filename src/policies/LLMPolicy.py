# TODO: LLM Reasoning Policy
# The LLM should take the intention that was communciated between agents in an intersection zone
# All other agents will yield to a leader vehicle

# LLM should consider: Intention, waiting time, (possibly distance and velocity)
# I.e, an emergency vehicle should have the highest priority

import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


class LLMPolicy:
    def __init__():
        # TODO: Possibly modify this
        content = """
There is another agent taking someone to a location. 
You will have a conversation until you determine whether you have more, less, or the same priority as them depending on the task you and they are performing. 
If you are doing the same task or have the same priority then say so. 
Do not include pleasantries and be concise. 
Once you have reached a consensus with the other agent, output the number 1 and nothing else. 
Remembering your task correctly is paramount!
""".strip()
        initial_msg = {"role": "system", "content": content}

    def query(self, role, prompt, persist=True):
        new_msg = {"role": role, "content": prompt}

        completion = openai.chat.completions.create(
            model="gpt-4o-mini", messages=self.messages + [new_msg]
        )

        if persist:
            self.messages.append(new_msg)
            self.messages.append(completion.choices[0].message)

        return completion.choices[0].message.content
