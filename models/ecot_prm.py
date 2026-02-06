import torch, numpy as np
from torch import nn
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


ECOT_TEMPLATE = """Task: {task}
Observation Analysis:
1. Gripper position: {gripper_pos}
2. Gripper state: {gripper_state}
3. Target object: {target_obj}
4. Current sub-task: {subtask}
5. Recent motion: {recent_motion}
6. Progress: {progress}

Based on this analysis, is progress being made toward the goal?"""


def obs_to_ecot_reasoning(obs, task_description, prev_obs=None, target_pos=None):
    "Convert LIBERO observation to ECoT reasoning dict"
    eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
    gripper_state = "open" if gripper_qpos[0] > 0.03 else "closed"
    pos_str = f"({eef_pos[0]:.2f}, {eef_pos[1]:.2f}, {eef_pos[2]:.2f})"
    
    recent_motion = "stationary"
    if prev_obs is not None:
        prev_eef = prev_obs.get("robot0_eef_pos", np.zeros(3))
        delta = eef_pos - prev_eef
        if np.linalg.norm(delta) > 0.01:
            directions = []
            if delta[0] > 0.005: directions.append("forward")
            elif delta[0] < -0.005: directions.append("backward")
            if delta[1] > 0.005: directions.append("right")
            elif delta[1] < -0.005: directions.append("left")
            if delta[2] > 0.005: directions.append("up")
            elif delta[2] < -0.005: directions.append("down")
            recent_motion = " and ".join(directions) if directions else "moving"
    
    progress = "unknown"
    if target_pos is not None and prev_obs is not None:
        prev_eef = prev_obs.get("robot0_eef_pos", np.zeros(3))
        prev_dist = np.linalg.norm(prev_eef - target_pos)
        curr_dist = np.linalg.norm(eef_pos - target_pos)
        if curr_dist < prev_dist - 0.005: progress = "approaching target"
        elif curr_dist > prev_dist + 0.005: progress = "moving away from target"
        else: progress = "maintaining distance"
    
    return dict(gripper_pos=pos_str, gripper_state=gripper_state, target_obj="[from task]",
                subtask="[inferred]", recent_motion=recent_motion, progress=progress)


def get_reward_rule_based(progress, gripper_state, task_phase="reach"):
    "Rule-based reward for baseline comparison"
    if task_phase == "reach":
        if progress == "approaching target" and gripper_state == "open": return 1
        if progress == "moving away from target": return 0
    elif task_phase == "grasp":
        if progress == "approaching target" and gripper_state == "closed": return 1
    elif task_phase == "place":
        if progress == "approaching target" and gripper_state == "closed": return 1
    return 0


class ECoTPRM(nn.Module):
    "Process Reward Model with Embodied Chain-of-Thought reasoning"
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.processor = AutoProcessor.from_pretrained(args.prm_model_name_or_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.prm_model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        if getattr(args, 'prm_checkpoint_path', None) is not None:
            self.model = PeftModel.from_pretrained(self.model, args.prm_checkpoint_path)
        self.model.eval()

    def build_prompt(self, task, image, obs, prev_obs=None, target_pos=None):
        "Build ECoT-style prompt for reward prediction"
        reasoning = obs_to_ecot_reasoning(obs, task, prev_obs, target_pos)
        text = ECOT_TEMPLATE.format(task=task, **reasoning)
        return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]

    @torch.no_grad()
    def get_reward(self, text_list, image_list, obs_list=None, prev_obs_list=None, target_pos_list=None):
        "Get reward scores with ECoT reasoning"
        if obs_list is None: obs_list = [{}] * len(text_list)
        if prev_obs_list is None: prev_obs_list = [None] * len(text_list)
        if target_pos_list is None: target_pos_list = [None] * len(text_list)
        
        messages = []
        for i, (text, img) in enumerate(zip(text_list, image_list)):
            task = text.split("What action should the robot take to ")[1].split("?")[0] if "What action" in text else text
            img_pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
            messages.append(self.build_prompt(task, img_pil, obs_list[i], prev_obs_list[i], target_pos_list[i]))
        
        prompts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        imgs = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in image_list]
        inputs = self.processor(text=prompts, images=imgs, padding=True, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=2)
        decoded = self.processor.batch_decode([o[len(i):] for i, o in zip(inputs.input_ids, outputs)], skip_special_tokens=True)
        return np.array([int(t.strip()) if t.strip().isdigit() else 0 for t in decoded])
