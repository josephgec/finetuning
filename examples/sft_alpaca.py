"""SFT on Alpaca-style dataset.

Usage:
    python examples/sft_alpaca.py --data path/to/alpaca.jsonl
"""

from __future__ import annotations

import argparse

from local_tinker import AdamParams, LoraConfig, ServiceClient
from local_tinker.losses import CrossEntropyLoss

from cookbook.supervised import SupervisedDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--data", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    client = ServiceClient()
    run = client.create_training_run(
        model=args.model,
        lora_config=LoraConfig(rank=16, alpha=32),
        quantize=args.quantize,
    )
    tc = run.training_client()

    dataset = SupervisedDataset.from_jsonl(args.data, run.tokenizer)
    loss_fn = CrossEntropyLoss()

    step = 0
    while step < args.steps:
        for batch in dataset.batch(args.batch_size):
            if step >= args.steps:
                break
            result = tc.forward_backward(batch, loss_fn)
            tc.optim_step(AdamParams(lr=args.lr))
            step += 1
            if step % 10 == 0:
                print(f"Step {step}/{args.steps} | Loss: {result.loss:.4f}")

    tc.save_weights("./sft_alpaca_output")
    print("Done!")


if __name__ == "__main__":
    main()
