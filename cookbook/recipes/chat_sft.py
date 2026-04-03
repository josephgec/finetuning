"""End-to-end chat SFT recipe.

Fine-tunes a model on a chat dataset using supervised learning.

Usage:
    python -m cookbook.recipes.chat_sft \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --data path/to/chat.jsonl \
        --steps 100
"""

from __future__ import annotations

import argparse
import json

from local_tinker import (
    AdamParams,
    LoraConfig,
    SamplingParams,
    ServiceClient,
)
from local_tinker.losses import CrossEntropyLoss

from cookbook.supervised import ChatDatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat SFT training")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--data", required=True, help="JSONL file with conversations")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()

    # Load conversations
    conversations = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    # Create run
    client = ServiceClient()
    run = client.create_training_run(
        model=args.model,
        lora_config=LoraConfig(rank=args.rank, alpha=args.rank * 2),
        quantize=args.quantize,
    )
    tc = run.training_client()

    # Build dataset
    builder = ChatDatasetBuilder(run.tokenizer)
    data = builder.build(conversations)

    # Training loop
    loss_fn = CrossEntropyLoss()
    step = 0
    while step < args.steps:
        for i in range(0, len(data), args.batch_size):
            if step >= args.steps:
                break
            batch = data[i : i + args.batch_size]
            result = tc.forward_backward(batch, loss_fn)
            tc.optim_step(AdamParams(lr=args.lr))
            step += 1
            print(f"Step {step}/{args.steps} | Loss: {result.loss:.4f}")

    # Save
    tc.save_weights("./chat_sft_output")
    print("Done! Weights saved to ./chat_sft_output")


if __name__ == "__main__":
    main()
