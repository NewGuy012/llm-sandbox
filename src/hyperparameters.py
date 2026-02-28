import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full", sql_output="native")

with app.setup:
    import marimo as mo
    import typer
    from rich import print
    from rich.pretty import pprint


@app.cell
def _():
    # if mo.app_meta().mode == "script":  
    #         app()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Device GPU Settings
    batch_size = 64
    block_size = 256
    n_layer = 6
    n_head = 6
    n_embd = 384
    dropout = 0.2
    learning_rate = 1e-3
    max_iters = 5000
    lr_decay_iters = 5000
    min_lr = 1e-4
    beta2 = 0.99
    warmup_iters = 100

    ### Device CPU Settings
    --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
    """)
    return


@app.function
# app = typer.Typer()

# @app.command()
def intialize_hyperparameters(
    batch_size: int = 4,
    block_size: int = 8,
    vocab_size = 50304,
    n_layer: int = 2,
    n_head: int = 2,
    n_embd: int = 64,
    dropout: float = 0.0,
    weight_decay: float = 1e-1,
    learning_rate: float = 3e-4,
    max_iters: int = 20,
    warmup_iters: int = 0,
    lr_decay_iters: int = 2000,
    min_lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.99,
    bias: bool = False,
    compile: bool = False,
    eval_iters: int = 20,
    eval_interval: int = 1,
    device: str = "cpu",
    init_from: str = "scratch"):

    config_dict = {
        "batch_size": batch_size,
        "block_size": block_size,
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "max_iters": max_iters,
        "warmup_iters": warmup_iters,
        "lr_decay_iters": lr_decay_iters,
        "min_lr": min_lr,
        "beta1": beta1,
        "beta2": beta2,
        "bias": bias,
        "compile": compile,
        "eval_iters": eval_iters,
        "eval_interval": eval_interval,
        "device": device,
        "init_from": init_from
    }

    print(config_dict)

    return config_dict


if __name__ == "__main__":
    app.run()
