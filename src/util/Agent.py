from .CollisionAvoidance import pastOrigin


class Agent:
    def __init__(self, agent):
        self.agent = agent
        self.id = agent.__hash__()
        self.position = agent.position
        self.velocity = agent.velocity
        self.pastIntersection = pastOrigin(agent.position, agent.direction)
        self.direction = agent.direction
        self.timeInZone = 0
        self.role = None
