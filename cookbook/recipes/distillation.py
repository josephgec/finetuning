"""Knowledge distillation recipe: train a small model to match a larger model.

Usage:
    python -m cookbook.recipes.distillation \
        --student meta-llama/Llama-3.2-1B-Instruct \
        --teacher meta-llama/Llama-3.1-8B-Instruct \
        --data path/to/prompts.jsonl \
        --steps 100
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from local_tinker import (
    AdamParams,
    LoraConfig,
    SamplingParams,
    ServiceClient,
)
from local_tinker.losses import CustomLoss


def kl_divergence_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor = None,
    temperature: float = 2.0,
    **kwargs: object,
) -> torch.Tensor:
    """KL divergence loss between student and teacher distributions."""
    if teacher_logits is None:
        return F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kl * (temperature ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge distillation")
    parser.add_argument("--student", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--teacher", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data", required=True, help="JSONL file with prompts")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    # Load prompts
    prompts: list[str] = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                prompts.append(row.get("prompt", row.get("text", "")))

    # Load student (with LoRA)
    student_client = ServiceClient()
    student_run = student_client.create_training_run(
        model=args.student,
        lora_config=LoraConfig(rank=16, alpha=32),
        quantize=args.quantize,
    )
    tc = student_run.training_client()
    sc = student_run.sampling_client()

    # Load teacher (frozen, just for inference)
    from transformers import AutoModelForCausalLM

    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    teacher_model.eval()

    tokenizer = student_run.tokenizer
    loss_fn = CustomLoss(kl_divergence_loss)

    step = 0
    while step < args.steps:
        for prompt in prompts:
            if step >= args.steps:
                break

            # Generate teacher completion
            teacher_response = sc.sample(
                prompt=__import__("local_tinker").ModelInput.from_text(prompt, tokenizer),
                params=SamplingParams(max_tokens=128, temperature=0.0),
            )
            full_text = prompt + teacher_response.text
            enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc.input_ids.to(tc._device)

            # Get teacher logits
            with torch.no_grad():
                teacher_out = teacher_model(input_ids=input_ids)

            # Student forward + backward with KL loss
            tc._model.train()
            student_out = tc._model(input_ids=input_ids)

            loss = kl_divergence_loss(
                student_out.logits,
                input_ids[0],
                teacher_logits=teacher_out.logits,
                temperature=args.temperature,
            )
            loss.backward()
            tc.optim_step(AdamParams(lr=args.lr))
            step += 1
            print(f"Step {step}/{args.steps} | KL Loss: {loss.item():.4f}")

    tc.save_weights("./distillation_output")
    print("Done! Weights saved to ./distillation_output")


if __name__ == "__main__":
    main()
