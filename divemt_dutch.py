import argparse
from datasets import load_from_disk
import sys
from transformers import (
    set_seed
)
import inseq
import os


def load_inseq_model(model_name, gradient):
    """ Load the inseq model to use"""
    print(f"Loading the inseq model...", file=sys.stderr)
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

    # Check if divemt_dutch_analyse folder exists, if not, create it
    if not os.path.exists("divemt_dutch_analyse"):
        print(f"'divemt_dutch_analyse' folder does not yet exist, creating it right away!", file=sys.stderr)
        os.makedirs("divemt_dutch_analyse")

    # Check if sentence subfolder folder exists, if not, create it
    if not os.path.exists(f"divemt_dutch_analyse/sen_{idx + 1}"):
        print(f"Sentence subfolder: 'sen_{idx + 1}' does not yet exist, creating it right away!", file=sys.stderr)
        os.makedirs(f"divemt_dutch_analyse/sen_{idx + 1}")

    # Create html and output it into predefined folder
    # Use aggregate("subwords") as we want the attributes for each word
    html = out.aggregate("subwords").show(display=False, return_html=True)
    with open(f"divemt_dutch_analyse/sen_{idx + 1}/sen_{idx + 1}.html", "w") as f:
        f.write(html)

    # Get all the words that are modified during post-editing \
    # and output them along with the target text
    words_to_analyse = [example["mt_tokens"][c] for c, tag in enumerate(example["mt_wmt22_qe"]) if tag == "BAD"]
    with open(f"divemt_dutch_analyse/sen_{idx + 1}/sen_{idx + 1}.txt", "w") as f:
        f.write(f"Source sen: {example['src_text']} \nMT sen: {example['mt_text']} \nTarget sen: {example['tgt_text']} \nModified words during post-editing: {str(words_to_analyse)}")


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
    print(f"Reloading 'divemt_data' from disk...", file=sys.stderr)
    dataset = load_from_disk("data/divemt_data")

    # Load model
    model = load_inseq_model(args.model, args.gradient)

    # Use the first 10 instances for now
    dataset.select(range(10)).map(
        comp_sal_scores,
        with_indices=True,
    )
