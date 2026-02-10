import numpy as np
from models.ecot_prm import obs_to_ecot_reasoning, get_reward_rule_based


class ECoTRewardWrapper:
    "Wrapper to provide dense ECoT-based rewards"
    def __init__(self, env, task_description, target_pos=None):
        self.env, self.task, self.target_pos = env, task_description, target_pos
        self.prev_obs = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_obs = obs
        return obs

    def step(self, action):
        obs, sparse_reward, done, info = self.env.step(action)
        reasoning = obs_to_ecot_reasoning(obs, self.task, self.prev_obs, self.target_pos)
        task_phase = self._infer_phase(obs)
        dense_reward = get_reward_rule_based(reasoning["progress"], reasoning["gripper_state"], task_phase)
        combined_reward = dense_reward * 0.1 + sparse_reward * 1.0  # sparse still dominates at episode end
        self.prev_obs = obs
        info["dense_reward"] = dense_reward
        info["sparse_reward"] = sparse_reward
        return obs, combined_reward, done, info

    def _infer_phase(self, obs):
        gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
        if gripper_qpos[0] > 0.03: return "reach"
        return "place"

    def __getattr__(self, name): return getattr(self.env, name)
