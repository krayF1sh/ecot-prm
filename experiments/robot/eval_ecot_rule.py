import argparse, json, os
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from envs.libero_env import LiberoVecEnv
from envs.ecot_reward_wrapper import ECoTRewardWrapper


def evaluate_rule_based(task_suite="libero_spatial", num_episodes=10):
    env = LiberoVecEnv(task_suite_name=task_suite, task_ids=[0], num_trials_per_task=num_episodes, seed=42)
    wrapped = ECoTRewardWrapper(env, env.task_descriptions[0])
    results = []
    for ep in range(num_episodes):
        obs = wrapped.reset()
        total_dense, total_sparse, steps = 0.0, 0.0, 0
        done = False
        while not done:
            action = env.action_space.sample()[None, :]
            obs, reward, done, truncated, info = wrapped.step(action)
            total_dense += float(info.get("dense_reward", 0))
            total_sparse += float(np.sum(info.get("sparse_reward", 0)))
            steps += 1
            done = done[0] if isinstance(done, (list, np.ndarray)) else done
            if steps >= 300: break
        success = info.get("success", [False])[0] if isinstance(info.get("success"), (list, np.ndarray)) else info.get("success", False)
        results.append(dict(episode=ep, steps=steps, dense=total_dense, sparse=total_sparse, success=bool(success)))
        print(f"Ep {ep}: steps={steps}, dense={total_dense:.2f}, sparse={total_sparse:.2f}, success={success}")
    env.close()
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
