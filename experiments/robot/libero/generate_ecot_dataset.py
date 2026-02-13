import argparse, json, os, h5py, numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from models.ecot_prm import obs_to_ecot_reasoning, ECOT_TEMPLATE


def extract_ecot_dataset(demo_file, task_description, output_dir, save_images=True):
    "Extract ECoT training samples from a single LIBERO demo HDF5 file"
    samples = []
    data = demo_file["data"]
    
    for demo_key in data.keys():
        demo = data[demo_key]
        obs_grp = demo["obs"]
        
        ee_states = obs_grp["ee_states"][()]
        gripper_states = obs_grp["gripper_states"][()]
        images = obs_grp["agentview_rgb"][()]
        rewards = demo["rewards"][()]
        
        success = rewards[-1] == 1
        n_steps = len(ee_states)
        
        for t in range(n_steps):
            obs = dict(robot0_eef_pos=ee_states[t, :3], robot0_gripper_qpos=gripper_states[t])
            prev_obs = dict(robot0_eef_pos=ee_states[t-1, :3], robot0_gripper_qpos=gripper_states[t-1]) if t > 0 else None
            
            reasoning = obs_to_ecot_reasoning(obs, task_description, prev_obs)
            ecot_text = ECOT_TEMPLATE.format(task=task_description, **reasoning)
            label = 1 if success else 0
            
            sample = dict(task=task_description, demo=demo_key, step=t, ecot_text=ecot_text, label=label)
            
            if save_images:
                img_path = output_dir / f"{demo_key}_step{t:04d}.png"
                Image.fromarray(images[t]).save(img_path)
                sample["image_path"] = str(img_path)
            
            samples.append(sample)
    
    return samples


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    
    all_samples = []
    hdf5_files = list(Path(args.libero_data_dir).glob("*_demo.hdf5"))
    
    for hdf5_path in tqdm(hdf5_files, desc="Processing tasks"):
        task_name = hdf5_path.stem.replace("_demo", "").replace("_", " ")
        with h5py.File(hdf5_path, "r") as f:
            samples = extract_ecot_dataset(f, task_name, output_dir / "images", save_images=args.save_images)
            all_samples.extend(samples)
    
    with open(output_dir / "ecot_dataset.json", "w") as f:
        json.dump(all_samples, f, indent=2)
    
    print(f"Extracted {len(all_samples)} samples to {output_dir}")
    print(f"Positive samples: {sum(s['label'] for s in all_samples)}")
    print(f"Negative samples: {sum(1-s['label'] for s in all_samples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_data_dir", type=str, required=True, help="Path to LIBERO HDF5 dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for ECoT dataset")
    parser.add_argument("--save_images", action="store_true", help="Save images as PNG files")
    args = parser.parse_args()
    main(args)