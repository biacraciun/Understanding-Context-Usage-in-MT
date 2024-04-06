import argparse
from datasets import Dataset, load_dataset
import sys
from transformers import (
    set_seed
)


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
    filtered_df = filtered_df[["item_id", "src_text", "mt_text", "tgt_text", "mt_tokens", "mt_wmt22_qe"]]
    print(f"{filtered_df.shape = }", file=sys.stderr)

    # Convert back to Dataset object
    # filtered_dataset = filtered_df.reset_format()
    return Dataset.from_pandas(filtered_df)


def get_context(examples, **fn_kwargs):
    """ This function retrieves the contexts for all selected examples in the data set. \
    These contexts (input and output) are used by PECoRe to determine contex usage."""

    lookup_df = fn_kwargs["lookup_data"]
    item_id = examples["item_id"].split("-")[-1]
    doc_id = int(item_id[:-1])
    sen_pos = int(item_id[-1])

    # If there is context, retrieve and return it
    if sen_pos > 1:
        src_con = ""
        tgt_con = ""
        for i in range(1, sen_pos):
            idx = lookup_df.loc[lookup_df["item_id"] == f"flores101-main-{doc_id}{i}"]
            src_con += " " + idx["src_text"].to_string(index=False)
            tgt_con += " " + idx["tgt_text"].to_string(index=False)
        # Add the context to the dataset
        return {
            "src_context": src_con[1:],
            "tgt_context": tgt_con[1:],
        }
    # If there is no context, return an empty string
    elif sen_pos == 1:
        # Add the empty context to the dataset
        return {
            "src_context": "",
            "tgt_context": "",
        }





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

    identifier = args.divEmt  # For example: "GroNLP/divemt"
    dataset = load_dataset(identifier)
    print(dataset, file=sys.stderr)
    print(f"{dataset['train'].format = }", file=sys.stderr)
    print(dataset["train"], file=sys.stderr)

    filtered_dataset = preprocess(dataset)
    print(filtered_dataset, file=sys.stderr)

    # Randomly select 50 examples
    use_dataset = filtered_dataset.shuffle(seed=random_seed).select(range(50))

    use_dataset = use_dataset.map(
        get_context,
        fn_kwargs={
            "lookup_data": filtered_dataset.to_pandas(),
        }
    )
    print(use_dataset, file=sys.stderr)

    # Save data set locally
    use_dataset.save_to_disk("data/divemt_data")
