import argparse, json, h5py, numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from models.ecot_prm import obs_to_ecot_reasoning, ECOT_TEMPLATE


def detect_milestones(ee_pos, gripper_states, vel_threshold=0.005, gripper_threshold=0.03, min_gap=10):
    n = len(ee_pos)
    milestones = []
    velocity = np.linalg.norm(np.diff(ee_pos, axis=0), axis=1)
    for t in range(1, n):
        prev_open = gripper_states[t-1, 0] > gripper_threshold
        curr_open = gripper_states[t, 0] > gripper_threshold
        if prev_open != curr_open: milestones.append(('gripper', t))
    for t in range(1, n-1):
        if velocity[t-1] < vel_threshold and velocity[t] < vel_threshold: milestones.append(('velocity', t))
    milestones.sort(key=lambda x: x[1])
    filtered = []
    for m_type, t in milestones:
        if not filtered or t - filtered[-1] >= min_gap: filtered.append(t)
    filtered.append(n-1)
    return sorted(set(filtered))


def compute_milestone_labels(n_steps, milestones):
    "Assign labels based on completed milestones"
    labels = np.zeros(n_steps)
    n_milestones = len(milestones)
    for i, m in enumerate(milestones):
        labels[m:] = (i + 1) / n_milestones
    return labels


def extract_ecot_dataset(hdf5_path, task_description, output_dir, save_images=True):
    samples = []
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        for demo_key in data.keys():
            demo = data[demo_key]
            obs_grp = demo["obs"]
            ee_pos = obs_grp["ee_pos"][()]
            gripper_states = obs_grp["gripper_states"][()]
            images = obs_grp["agentview_rgb"][()]
            rewards = demo["rewards"][()]
            success = rewards[-1] == 1
            if not success: continue
            n_steps = len(ee_pos)
            milestones = detect_milestones(ee_pos, gripper_states)
            labels = compute_milestone_labels(n_steps, milestones)
            for t in range(n_steps):
                obs = dict(robot0_eef_pos=ee_pos[t], robot0_gripper_qpos=gripper_states[t])
                prev_obs = dict(robot0_eef_pos=ee_pos[t-1], robot0_gripper_qpos=gripper_states[t-1]) if t > 0 else None
                reasoning = obs_to_ecot_reasoning(obs, task_description, prev_obs)
                ecot_text = ECOT_TEMPLATE.format(task=task_description, **reasoning)
                sample = dict(task=task_description, demo=demo_key, step=t, ecot_text=ecot_text, label=float(labels[t]), is_milestone=t in milestones)
                if save_images:
                    img_path = output_dir / "images" / f"{Path(hdf5_path).stem}_{demo_key}_step{t:04d}.png"
                    Image.fromarray(images[t]).save(img_path)
                    sample["image_path"] = str(img_path.relative_to(output_dir))
                samples.append(sample)
    return samples


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images: (output_dir / "images").mkdir(exist_ok=True)
    all_samples = []
    hdf5_files = list(Path(args.libero_data_dir).glob("*_demo.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files")
    for hdf5_path in tqdm(hdf5_files, desc="Processing tasks"):
        task_name = hdf5_path.stem.replace("_demo", "").replace("_", " ")
        samples = extract_ecot_dataset(hdf5_path, task_name, output_dir, save_images=args.save_images)
        all_samples.extend(samples)
    with open(output_dir / "ecot_dataset.json", "w") as f: json.dump(all_samples, f, indent=2)
    n_milestones = sum(1 for s in all_samples if s["is_milestone"])
    print(f"Extracted {len(all_samples)} samples ({n_milestones} milestones)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_images", action="store_true")
    args = parser.parse_args()
    main(args)