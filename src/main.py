import gymnasium as gym
import highway_env
from MultiAgentWrapper import MultiAgentWrapper
import imageio
import random

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    ap.add_argument("-t", "--test", action="store_true", help="Run tests")
    ap.add_argument(
        "-n",
        "--num-agents",
        type=int,
        default=2,
        help="Number of agents in the environment",
    )
    ap.add_argument(
        "--max-iter", type=int, default=100, help="Max number of iterations"
    )
    args = ap.parse_args()

    env = MultiAgentWrapper(num_agents=args.num_agents)
    obs, info = env.reset()

    frames = []

    for iteration in range(args.max_iter):
        # actions = [predict(o) for o in obs]
        if iteration < 3:
            actions = ([-0.3, 0], [0, 0])
        elif iteration < 6:
            actions = ([0.2, 0], [0, 0])
        else:
            actions = ([0, 0], [0, 0])
        assert len(actions) == env.num_agents
        _obs, rewards, done, truncated, info = env.step(actions)
        env.step(actions)
        frames.append(env.render())
        obs = _obs
    env.close()

    if args.test:
        imageio.mimsave(f"res/out.mp4", frames, fps=2)


if __name__ == "__main__":
    main()
