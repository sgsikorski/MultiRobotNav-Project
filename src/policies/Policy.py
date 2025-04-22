from .LLMPolicy import LLMPolicy
from .BaseAuction import AuctionPolicy


class Policy:
    def __init__(self, use_llm=False):
        self.use_llm = use_llm
        self.policy = LLMPolicy() if use_llm else AuctionPolicy()
        self.policy = AuctionPolicy()

    def get_leader_followers(self, agents):
        if not hasattr(self.policy, "get_leader_followers"):
            raise NotImplementedError(
                "get_leader_followers for the designated policy is not implemented yet"
            )
        return self.policy.get_leader_followers(agents)
