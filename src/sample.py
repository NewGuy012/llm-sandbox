import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full", sql_output="native")

with app.setup:
    import torch
    import tiktoken
    import marimo as mo
    from dataclasses import dataclass


@app.class_definition
@dataclass
class SampleConfig:
    start: str = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples: int = 1 # number of samples to draw
    max_new_tokens: int = 500 # number of tokens generated in each sample
    temperature: float = 0.95 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k: int = 200


@app.function
def sample(hyper_config, sample_config, model):
    device  = hyper_config["device"]
    compile = hyper_config["compile"]

    start = sample_config.start
    num_samples = sample_config.num_samples
    max_new_tokens = sample_config.max_new_tokens
    temperature = sample_config.temperature
    top_k = sample_config.top_k

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # ok let's assume gpt-2 encodings by default
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    if start.startswith("FILE:"):
        with open(start[5:], "r", encoding="utf-8") as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')


if __name__ == "__main__":
    app.run()
