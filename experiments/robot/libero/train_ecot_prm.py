import argparse, json, torch, os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType


class ECoTDataset(Dataset):
    def __init__(self, data_path, processor, max_samples=None):
        with open(data_path / "ecot_dataset.json") as f: self.samples = json.load(f)
        if max_samples: self.samples = self.samples[:max_samples]
        self.data_path = data_path
        self.processor = processor

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(self.data_path / s["image_path"]).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": s["ecot_text"]}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[img], return_tensors="pt", padding=True)
        return dict(input_ids=inputs.input_ids.squeeze(0), attention_mask=inputs.attention_mask.squeeze(0), 
                    pixel_values=inputs.pixel_values.squeeze(0), labels=torch.tensor(s["label"], dtype=torch.float32))


class ECoTPRMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :2]
        probs = torch.softmax(logits, dim=-1)[:, 1]
        loss = torch.nn.functional.mse_loss(probs, labels)
        return (loss, outputs) if return_outputs else loss


def main(args):
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha, 
                              lora_dropout=0.1, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    train_dataset = ECoTDataset(Path(args.data_dir), processor, max_samples=args.max_samples)
    training_args = TrainingArguments(output_dir=args.output_dir, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, bf16=True, logging_steps=10, save_steps=500,
        save_total_limit=2, dataloader_num_workers=4, report_to="tensorboard")
    trainer = ECoTPRMTrainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    model.save_pretrained(args.output_dir / "final_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/ecot_prm")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    main(args)