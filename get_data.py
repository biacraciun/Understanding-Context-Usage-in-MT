import argparse
from datasets import Dataset, load_dataset
import sys
import numpy as np
import pandas as pd
from transformers import (
    set_seed
)
import inseq


def preprocess(dataset):
    """ This function filters the data set by retrieving \
    the Dutch translations by mBART and only maintaining \
    the relevant columns: source text, translated text, \
    target text, translated tokens in a list and \
    whether a token is kept or edited during post-editing."""

    # Convert to Pandas DataFrame (set_format() did not work)
    # df = dataset["train"].set_format("pandas")
    df = dataset["train"].to_pandas()
    del dataset
    print(f"{df.shape = }", file=sys.stderr)

    # Filter data to extract the Dutch (lang_id == "nld") translations of mBART (task_type == "pe2")
    filtered_df = df.loc[(df["lang_id"] == "nld") & (df["task_type"] == "pe2")]

    # Only maintain relevant information
    filtered_df = filtered_df[["src_text", "mt_text", "tgt_text", "mt_tokens", "mt_wmt22_qe"]]
    print(f"{filtered_df.shape = }", file=sys.stderr)

    # Convert back to Dataset object
    # filtered_dataset = filtered_df.reset_format()
    return Dataset.from_pandas(filtered_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--divEmt",
        "-de",
        help="Name of the divEMT data set on hugging face. Default: 'GroNLP/divemt'",
        default="GroNLP/divemt",
        type=str,
    )
    #parser.add_argument(
    #    "--output_file_path",
    #    "-out",
    #    required=True,
    #    help="Path where to save the output file (synthetic data set).",
    #    type=str,
    #)
    args = parser.parse_args()
    random_seed = 0
    set_seed(random_seed)

    identifier = args.divEmt  # For example: "divemt"
    dataset = load_dataset(identifier)
    print(dataset, file=sys.stderr)
    print(f"{dataset['train'].format = }", file=sys.stderr)
    print(dataset["train"], file=sys.stderr)

    filtered_dataset = preprocess(dataset)
    print(filtered_dataset, file=sys.stderr)

    # Randomly select 50 examples
    use_dataset = filtered_dataset.shuffle(seed=random_seed).select(range(50))

    # Save data set locally
    use_dataset.save_to_disk("data")
