import argparse
from datasets import Dataset, load_dataset, load_from_disk
import sys
import numpy as np
import pandas as pd
from transformers import (
    set_seed
)
import inseq


def load_inseq_model(model_name, gradient):
    """ Load the inseq model to use"""
    return inseq.load_model(
        model_name,
        gradient,
        # These arguments are used by Huggingface to specify the source and target languages used by the model
        tokenizer_kwargs={"src_lang": src_lang, "tgt_lang": tgt_lang},
    )


def comp_sal_scores(example, idx):
    """ This function calculates the salient (importance) score between \
    two provided sentences and outputs the HTML file to a prefixed folder. \
    Moreover, it outputs the modified words during post-editing for further analysis."""

    # Get attributes
    out = model.attribute(
        example["src_text"],
        example["mt_text"],
        attribute_target=False,
        step_scores=["probability"],
    )

    # Create html and output it into predefined folder
    # Use aggregate("subwords") as we want the attributes for each word
    html = out.aggregate("subwords").show(display=False, return_html=True)
    with open(f"html/sen_{idx}.html", "w") as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument(
    #    "--divEmt",
    #    "-de",
    #    help="Name of the divEMT data set on hugging face. Default: 'GroNLP/divemt'",
    #    default="GroNLP/divemt",
    #    type=str,
    #)
    #parser.add_argument(
    #    "--output_file_path",
    #    "-out",
    #    required=True,
    #    help="Path where to save the output file (synthetic data set).",
    #    type=str,
    #)
    parser.add_argument(
        "--source_language",
        "-src",
        help="Language code of the source language. Default: 'en_XX'",
        default="en_XX",
        type=str,
    )
    parser.add_argument(
        "--target_language",
        "-tgt_lang",
        help="Language code of the target language. Default: 'nl_XX'",
        default="nl_XX",
        type=str,
    )
    parser.add_argument(
        "--gradient",
        "-grad",
        help="Gradient-based attribution to use. Default: 'saliency'",
        default="saliency",
        type=str,
    )
    parser.add_argument(
        "--model",
        "-m",
        help="The model used to translate. Default: 'facebook/mbart-large-50-one-to-many-mmt'",
        default="facebook/mbart-large-50-one-to-many-mmt",
        type=str,
    )
    args = parser.parse_args()
    random_seed = 0
    set_seed(random_seed)
    src_lang = args.source_language
    tgt_lang = args.target_language

    # Reload data from disk
    dataset = load_from_disk("data/divemt_data")

    # Load model
    model = load_inseq_model(args.model, args.gradient)

    # Use the first 10 instances for now
    dataset.select(range(5)).map(
        comp_sal_scores,
        with_indices=True,
    )
