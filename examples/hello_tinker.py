"""Minimal end-to-end smoke test for Local Tinker.

Loads a small model, runs one SFT training step, then samples a completion.
"""

from local_tinker import (
    AdamParams,
    Datum,
    LoraConfig,
    ModelInput,
    SamplingParams,
    ServiceClient,
)
from local_tinker.losses import CrossEntropyLoss


def main() -> None:
    # 1. Create a training run
    client = ServiceClient()
    run = client.create_training_run(
        model="meta-llama/Llama-3.2-1B-Instruct",
        lora_config=LoraConfig(rank=16, alpha=32, target_modules=["q_proj", "v_proj"]),
    )
    tc = run.training_client()
    sc = run.sampling_client()
    tokenizer = run.tokenizer

    print(f"Model: {run.model_name}")
    print(f"Device: {run.device}")

    # 2. One SFT training step
    text = "The capital of France is Paris."
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded.input_ids[0].tolist()

    result = tc.forward_backward(
        data=[Datum(input_ids=ids, labels=ids)],
        loss_fn=CrossEntropyLoss(),
    )
    print(f"Loss: {result.loss:.4f}  |  Tokens: {result.num_tokens}")

    tc.optim_step(AdamParams(lr=1e-4))
    print(f"Step: {tc.get_step()}")

    # 3. Sample a completion
    output = sc.sample(
        prompt=ModelInput.from_text("The capital of France is", tokenizer),
        params=SamplingParams(max_tokens=20, temperature=0.7),
    )
    print(f"Generated: {output.text}")


if __name__ == "__main__":
    main()
