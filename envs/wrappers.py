import os
import numpy as np
import gymnasium as gym
from experiments.robot.libero.libero_utils import save_rollout_video


class VideoWrapper(gym.Wrapper):
    def __init__(self, env, save_dir: str = "video", save_freq: int = 1):
        super().__init__(env)
        self.env = env
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.frames = []
        self.episode_count = 0
        
        os.makedirs(save_dir, exist_ok=True)
    
    def reset(self, **kwargs):
        self.frames = []
        self.current_step = 0
        obs, info = self.env.reset(**kwargs)
        
        if self.episode_count % self.save_freq == 0:
            img = obs.pixel_values[0] if isinstance(obs.pixel_values, list) else obs.pixel_values
            self.frames.append(img)
        
        return obs, info
    
    def step(self, actions, **kwargs):
        obs, rewards, dones, info = self.env.step(actions)
        
        if self.episode_count % self.save_freq == 0:
            img = obs.pixel_values[0] if isinstance(obs.pixel_values, list) else obs.pixel_values
            self.frames.append(img)
        
        if np.any(dones):
            if self.episode_count % self.save_freq == 0:
                self._save_video(rewards, dones)
            self.episode_count += 1
        
        self.current_step = 1
            
        return obs, rewards, dones, info
    
    def _save_video(self, rewards=None, dones=None):
        """Save the collected frames as a video file."""
        if not self.frames:
            return
        success = False
        if rewards is not None:
            if isinstance(rewards, (list, np.ndarray)):
                success = np.any(np.array(rewards) > 0)
            else:
                success = rewards > 0
        
        task_description = self.env.task_descriptions[0]
        processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
            
        mp4_path = os.path.join(
            self.save_dir, 
            f"episode={self.episode_count}--success={success}--task={processed_task_description}.mp4"
        )
        save_rollout_video(
            self.frames, 
            self.episode_count, 
            success=success,
            task_description=str(task_description),
            log_file=None,
            mp4_path=mp4_path,
        )
        print(f"Video saved to: {mp4_path}")
        self.frames = []
    
    def close(self):
        self.env.close()


class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, temp: float = 1.0, min_prob: float = 0.1, window_size: int = 5):
        super().__init__(env)
        self.env = env
        self.temp = temp
        self.min_prob = min_prob
        self.window_size = window_size
        self.success_tracker = {}
        self.state_history = {}
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, actions, **kwargs):
        return self.env.step(actions, **kwargs)
    
    def update_success(self, task_id: int, state_id: int, success: bool):
        if task_id not in self.success_tracker:
            self.success_tracker[task_id] = {}
        
        if state_id not in self.success_tracker[task_id]:
            self.success_tracker[task_id][state_id] = []
        
        self.success_tracker[task_id][state_id].append(1.0 if success else 0.0)
        
        if len(self.success_tracker[task_id][state_id]) > self.window_size:
            self.success_tracker[task_id][state_id].pop(0)
    
    def get_success_rate(self, task_id: int, state_id: int) -> float:
        if task_id not in self.success_tracker or state_id not in self.success_tracker[task_id]:
            return 0.0
        
        history = self.success_tracker[task_id][state_id]
        if not history:
            return 0.0
        
        return sum(history) / len(history)
    
    def sample_state(self, task_id: int, n_states: int) -> int:
        if task_id not in self.success_tracker:
            return np.random.randint(0, n_states)
        
        weights = []
        for state_id in range(n_states):
            success_rate = self.get_success_rate(task_id, state_id)
            weight = (1.0 - success_rate + 1e-9) ** (1.0 / self.temp)
            weights.append(weight)
        
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        return np.random.choice(len(weights), p=probabilities)
    
    def close(self):
        self.env.close()