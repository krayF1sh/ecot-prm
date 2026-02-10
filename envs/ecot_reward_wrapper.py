import numpy as np
from models.ecot_prm import obs_to_ecot_reasoning, get_reward_rule_based, get_target_pos_from_task


class ECoTRewardWrapper:
    "Wrapper to provide dense ECoT-based rewards"
    def __init__(self, env, task_description, target_pos=None):
        self.env, self.task, self.target_pos = env, task_description, target_pos
        self.prev_obs = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_obs = None
        return obs

    def step(self, action):
        obs, sparse_reward, done, truncated, info = self.env.step(action)
        raw_obs = info.get("raw_obs", [{}])[0]
        self.step_counts = getattr(self, 'step_counts', 0) + 1
        target_pos = get_target_pos_from_task(self.task, raw_obs)
        prev_raw = self.prev_obs if self.prev_obs else None
        reasoning = obs_to_ecot_reasoning(raw_obs, self.task, prev_raw, target_pos)
        # if self.step_counts <= 3: print(f"DEBUG target_pos: {target_pos}, progress: {reasoning['progress']}")
        task_phase = self._infer_phase(raw_obs)
        dense_reward = get_reward_rule_based(reasoning["progress"], reasoning["gripper_state"], task_phase)
        combined_reward = dense_reward * 0.1 + sparse_reward
        self.prev_obs = raw_obs
        info["dense_reward"] = dense_reward
        info["sparse_reward"] = sparse_reward
        return obs, combined_reward, done, truncated, info

    # def step(self, action):
    #     obs, sparse_reward, done, truncated, info = self.env.step(action)
    #     self.step_counts += 1
    #     dense_reward = 0.1 if self.step_counts < 50 else 0.0
    #     combined_reward = sparse_reward + dense_reward * 0.1
    #     info["dense_reward"] = dense_reward
    #     info["sparse_reward"] = float(np.sum(sparse_reward))
    #     return obs, combined_reward, done, truncated, info

    def _infer_phase(self, obs):
        gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
        return "reach" if gripper_qpos[0] > 0.03 else "place"

    def __getattr__(self, name): return getattr(self.env, name)
