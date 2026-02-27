import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full", sql_output="native")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    from tokenizer import download, tokenize
    from hyperparameters import intialize_hyperparameters
    from train import initialize_model, train
    from sample import SampleConfig, sample

    return initialize_model, intialize_hyperparameters, train


@app.cell
def _(initialize_model, intialize_hyperparameters, train):
    config = intialize_hyperparameters(
        device = "cpu",
        batch_size = 4,
        block_size = 8,
        vocab_size = 50304,
        n_layer=4,
        n_head=4,
        n_embd=128,
        max_iters = 20,
        learning_rate = 3e-4,
        eval_iters = 10,
        eval_interval = 1,
        bias=False)

    model, optimizer = initialize_model(config)

    train(config, model, optimizer)
    return


@app.cell
def _():
    # ### Tokenize ###
    # file_name = "train-validation.safetensors"
    # root_file = Path(__file__).parent
    # save_path = root_file / "data" / file_name

    # if not save_path.exists():
    #     ds = download()
    #     ds_tok = tokenize(ds)
    #     save(ds, save_path)
    return


@app.cell
def _():
    # ### Hyperpameters ###
    # hyper_config = intialize_hyperparameters()
    return


@app.cell
def _():
    # ### Model ###
    # model, optimizer = initialize_model(hyper_config)
    return


@app.cell
def _():
    # ### Train ###
    # train(hyper_config, model, optimizer)
    return


@app.cell
def _():
    # ### Sample ###
    # sample_config = SampleConfig()
    # print(sample_config)
    # sample(hyper_config, sample_config, model)
    return


if __name__ == "__main__":
    app.run()
