"""DPO preference tuning recipe.

Usage:
    python -m cookbook.recipes.dpo_preference \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --data path/to/preferences.jsonl \
        --steps 100
"""

from __future__ import annotations

import argparse

from local_tinker import (
    AdamParams,
    LoraConfig,
    ServiceClient,
)
from local_tinker.losses import DPOLoss

from cookbook.preference import PreferenceDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO preference tuning")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--data", required=True, help="JSONL file with preference pairs")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    client = ServiceClient()
    run = client.create_training_run(
        model=args.model,
        lora_config=LoraConfig(rank=16, alpha=32),
        quantize=args.quantize,
    )
    tc = run.training_client()

    dataset = PreferenceDataset.from_jsonl(
        args.data, run.tokenizer
    )

    loss_fn = DPOLoss(beta=args.beta)
    step = 0

    while step < args.steps:
        for batch in dataset.batch(args.batch_size):
            if step >= args.steps:
                break

            # Get reference log-probs for chosen and rejected
            chosen_data = [pair[0] for pair in batch]
            rejected_data = [pair[1] for pair in batch]

            ref_chosen = tc.get_reference_log_probs(chosen_data)
            ref_rejected = tc.get_reference_log_probs(rejected_data)

            # Combine chosen + rejected for the forward pass
            combined = chosen_data + rejected_data

            # Custom forward pass with DPO loss kwargs
            tc._model.train()
            input_ids, labels, attention_mask = tc._collate(combined)
            outputs = tc._model(input_ids=input_ids, attention_mask=attention_mask)

            loss = loss_fn.compute(
                outputs.logits,
                labels,
                ref_chosen_log_probs=ref_chosen,
                ref_rejected_log_probs=ref_rejected,
            )
            loss.backward()
            tc.optim_step(AdamParams(lr=args.lr))
            step += 1
            print(f"Step {step}/{args.steps} | DPO Loss: {loss.item():.4f}")

    tc.save_weights("./dpo_output")
    print("Done! Weights saved to ./dpo_output")


if __name__ == "__main__":
    main()
