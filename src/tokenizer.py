import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full", sql_output="native")

with app.setup:
    import torch
    import tiktoken
    import marimo as mo
    from pathlib import Path
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from safetensors.torch import save_file


@app.function
def download():
    # jchwenger/tiny_shakespeare 
    dataset = load_dataset("Trelis/tiny-shakespeare")
    dataset = dataset.rename_column("Text", "text")

    return dataset


@app.function
def tokenize(ds):
    enc = tiktoken.get_encoding("gpt2")

    # Load the standard Python tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def tokenization(ds):
        token_dict = tokenizer(ds["text"])

        input_ids = token_dict["input_ids"]

        for input_id in input_ids:
            input_id.append(enc.eot_token)

        token_dict["input_ids"] = input_ids

        return token_dict

     # Batch tokenize
    tokenized = ds.map(
        tokenization,
        remove_columns=['text'],
        desc="HF tokenizer",
        batched=True)

    return tokenized


@app.function
def tokenize_tiktoken(ds):
    enc = tiktoken.get_encoding("gpt2")

    def process(ds):
        ids = enc.encode_ordinary(ds['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = ds.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing the splits"
    )

    return tokenized


@app.function
def save(ds, save_path):
    ds.set_format(type='torch', columns=['input_ids'])
    
    ds_train = torch.cat(ds["train"]["input_ids"][:])
    ds_val = torch.cat(ds["test"]["input_ids"][:])

    tensors = {
        "train": ds_train,
        "validation": ds_val
    }

    save_file(tensors, save_path)

    return ds


if __name__ == "__main__":
    app.run()
