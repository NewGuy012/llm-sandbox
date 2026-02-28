import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full", sql_output="native")

with app.setup:
    import math
    import pickle
    import torch
    import marimo as mo
    from model import GPTConfig, GPT
    from pathlib import Path
    from safetensors import safe_open

    from torch.utils.data import TensorDataset, DataLoader


@app.function
def get_batch_loader(config, split):
    batch_size = config["batch_size"]
    block_size = config["block_size"]
    device  = config["device"]
    
    file_name = "train-validation.safetensors"
    root_file = Path(__file__).parent.parent
    file_path = root_file / "data" / file_name
    
    with safe_open(file_path, framework="pt", device="cpu") as f:
            # Access tensors by name
            data = f.get_tensor(split)
    
    data_length = len(data)
    quotient = (data_length // block_size)-1
    truncate_length = block_size * quotient
    
    x = data[:truncate_length].view(-1, block_size)
    y = data[1:truncate_length+1].view(-1, block_size)
    
    dataset = TensorDataset(x, y)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


@app.function
def get_batch_slice(config, split):
    batch_size = config["batch_size"]
    block_size = config["block_size"]
    device  = config["device"]

    file_name = "train-validation.safetensors"
    root_file = Path(__file__).parent.parent
    file_path = root_file / "data" / file_name

    with safe_open(file_path, framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice(split)
        data = tensor_slice[:batch_size*block_size + 1]

    x = data[:-1].view(batch_size, block_size)
    y = data[1:].view(batch_size, block_size)

    x, y = x.to(device), y.to(device)

    return x, y


@app.function
def get_batch_random(config, split):
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    device  = config["device"]

    file_name = "train-validation.safetensors"
    root_file = Path(__file__).parent.parent
    file_path = root_file / "data" / file_name

    with safe_open(file_path, framework="pt", device="cpu") as f:
        # Access tensors by name
        data = f.get_tensor(split)

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y


@app.function
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(config, model):
    eval_iters = config["eval_iters"]

    out = {}
    model.eval()
    for split in ["train", "validation"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_random(config, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@app.function
# learning rate decay scheduler (cosine with warmup)
def get_lr(config, it):
    learning_rate = config["learning_rate"]
    warmup_iters = config["warmup_iters"]
    lr_decay_iters = config["lr_decay_iters"]
    min_lr = config["min_lr"]

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)

    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr

    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1

    return min_lr + coeff * (learning_rate - min_lr)


@app.function
def initialize_model(config):
    vocab_size = config["vocab_size"]
    block_size = config["block_size"]
    n_layer = config["n_layer"]
    n_head = config["n_head"]
    n_embd = config["n_embd"]
    dropout = config["dropout"]
    bias = config["bias"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    device = config["device"]
    compile = config["compile"]
    init_from = config["init_from"]

    if init_from == 'scratch':
        # Initialize model
        print("Initializing a new model from scratch")
        print(f"defaulting to vocab_size of GPT-2 to {vocab_size}")

        model_args = dict(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            bias=bias,
            dropout=dropout)
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    elif init_from.startswith('gpt2'):
        # initialize from OpenAI GPT-2 weights
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")

        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        # model_args["block_size"] = block_size # so that the checkpoint will have the right value

    model.to(device)

    # Initialize optimizer
    optimizer = model.configure_optimizers(
        weight_decay,
        learning_rate,
        (beta1, beta2),
        device)

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model) # requires PyTorch 2.0

    return model, optimizer


@app.function
def train_random_batches(config, model, optimizer):
    max_iters = config["max_iters"]
    lr_decay_iters = config["lr_decay_iters"]
    eval_interval = config["eval_interval"]

    # training loop
    for iter_num in range(max_iters):
        # determine and set the learning rate for this iteration
        if lr_decay_iters > max_iters:
            lr = get_lr(config, iter_num)

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(config, model)
            print(f"step {iter_num}: train loss {losses["train"]:.4f}, val loss {losses["validation"]:.4f}")

        # sample a batch of data
        X, Y = get_batch_random(config, "train")

        # evaluate the loss
        logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


@app.function
def train_sequential_batches(config, model, optimizer):
    max_iters = config["max_iters"]
    lr_decay_iters = config["lr_decay_iters"]
    eval_interval = config["eval_interval"]

    loader = get_batch_loader(config, "train")
    loader_iter = iter(loader)
    
    # training loop
    for iter_num in range(max_iters):
        # determine and set the learning rate for this iteration
        if lr_decay_iters > max_iters:
            lr = get_lr(config, iter_num)

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(config, model)
            print(f"step {iter_num}: train loss {losses["train"]:.4f}, val loss {losses["validation"]:.4f}")

        # sample a batch of data
        X, Y  = next(loader_iter)

        # evaluate the loss
        logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    app.run()
