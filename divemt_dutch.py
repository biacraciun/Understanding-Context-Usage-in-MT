import argparse
from datasets import load_from_disk
import sys
from transformers import (
    set_seed
)
import os
import inseq
from inseq.commands.attribute_context.attribute_context import attribute_context_with_model, AttributeContextArgs


def load_inseq_model(model_name, gradient):
    """ Load the inseq model to use"""
    print(f"Loading the inseq model...", file=sys.stderr)
    return inseq.load_model(
        model_name,
        gradient,
        # These arguments are used by Huggingface to specify the source and target languages used by the model
        tokenizer_kwargs={"src_lang": src_lang, "tgt_lang": tgt_lang},
    )


def get_pecore_args(sentence_eng_Latn, context_eng_Latn, context_target_language,
               sub_directory_name):
    """
    This function uses the PECORE model to attribute the context of a sentence in a target language.
    The code was adapted from the PECORE demo. https://huggingface.co/spaces/gsarti/pecore
    Args:
        sentence_eng_Latn (str): The current sentence in English.
        context_eng_Latn (str): The context in English.
        context_target_language (str): The context in the target language.
        sub_directory_name (path): The subdirectory to save the files to.
        (global variable) tgt_lang (str): The language code of the target language.
    Returns:
        The PECoRe args
    """

    # Set the arguments for the PECoRe model
    pecore_args = AttributeContextArgs(
        model_name_or_path=args.model,
        attribution_method=args.gradient,
        attributed_fn="probability",
        context_sensitivity_metric="kl_divergence",
        context_sensitivity_std_threshold=0,
        attribution_std_threshold=2,
        attribution_topk=5,
        input_current_text=sentence_eng_Latn,
        input_context_text=context_eng_Latn,
        contextless_input_current_text="""{current}""",
        input_template="""{context} {current}""",
        output_context_text=context_target_language,
        contextless_output_current_text="""{current}""",
        output_template="{context} {current}",
        save_path=sub_directory_name + "_pec.json",
        viz_path=sub_directory_name + "_pec.html",
        tokenizer_kwargs={'src_lang': 'eng_Latn', 'tgt_lang': tgt_lang},
    )

    return pecore_args


def comp_sal_scores(example, idx):
    """ This function calculates the salient (importance) score between \
    two provided sentences and outputs the HTML file to a prefixed folder. \
    Additionally, it outputs the PECoRE context attribution to the same prefixed folder. \
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
    # and output them along with the source text, predicted text and \
    # target text
    words_to_analyse = [example["mt_tokens"][c] for c, tag in enumerate(example["mt_wmt22_qe"]) if tag == "BAD"]
    with open(f"divemt_dutch_analyse/sen_{idx + 1}/sen_{idx + 1}.txt", "w") as f:
        f.write(f"Source sen: {example['src_text']} \nMT sen: {example['mt_text']} \nTarget sen: {example['tgt_text']} \nModified words during post-editing: {str(words_to_analyse)}")

    # Run PECoRe and output the overall context attribution process
    try:
        pecore_args = get_pecore_args(
            sentence_eng_Latn=example["src_text"],
            context_eng_Latn=example["src_context"],
            context_target_language=example["tgt_context"],
            sub_directory_name=f"divemt_dutch_analyse/sen_{idx + 1}/sen_{idx + 1}")

        return attribute_context_with_model(pecore_args, model)
    except ValueError:
        print(f"Sentence {idx + 1} (example {idx}) has no context as it is the first sentence. "
              f"Hence, PECoRe cannot be utilised.", file=sys.stderr)


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
