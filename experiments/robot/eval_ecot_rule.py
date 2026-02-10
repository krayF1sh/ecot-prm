import argparse, json
import numpy as np
from pathlib import Path
from envs.libero_env import LiberoEnv
from envs.ecot_reward_wrapper import ECoTRewardWrapper


def evaluate_rule_based(task_suite="libero_spatial", num_episodes=10):
    env = LiberoEnv(task_suite=task_suite, seed=42)
    results = []
    for ep in range(num_episodes):
        obs = env.reset()
        wrapped = ECoTRewardWrapper(env, env.task_description)
        total_dense, total_sparse, steps = 0, 0, 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = wrapped.step(action)
            total_dense += info["dense_reward"]
            total_sparse += info["sparse_reward"]
            steps += 1
        results.append(dict(episode=ep, steps=steps, dense=total_dense, sparse=total_sparse, success=info.get("success", False)))
        print(f"Ep {ep}: steps={steps}, dense={total_dense:.2f}, sparse={total_sparse:.2f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite", default="libero_spatial")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--output", default="results/ecot_rule_eval.json")
    args = parser.parse_args()
    results = evaluate_rule_based(args.task_suite, args.num_episodes)
    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f: json.dump(results, f, indent=2)
