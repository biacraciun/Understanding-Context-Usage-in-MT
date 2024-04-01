import inseq
from inseq.commands.attribute_context.attribute_context import (
    AttributeContextArgs,
    attribute_context_with_model,
)

def load_model(language_code):
    
    # Load the model speecific to the target language
    inseq_model = inseq.load_model(
        "facebook/nllb-200-distilled-600M",
        "saliency",
        tokenizer_kwargs={'src_lang': 'eng_Latn', 'tgt_lang': language_code},
    )
    
    return inseq_model


def use_pecore(sentence_eng_Latn, context_eng_Latn, context_target_language, language_code, inseq_model, sub_directory_name):
    """
    This function uses the PECORE model to attribute the context of a sentence in a target language.
    The code was adapted from the PECORE demo. https://huggingface.co/spaces/gsarti/pecore
    Args:
        sentence_eng_Latn (str): The current sentence in English.
        context_eng_Latn (str): The context in English.
        context_target_language (str): The context in the target language.
        language_code (str): The language code of the target language.
    Returns:
        Output of the overall context attribution process.
    """
    # create directory for saving the output
        

    # Set the arguments for the PECORE model
    pecore_args = AttributeContextArgs(
        model_name_or_path="facebook/nllb-200-distilled-600M",
        attribution_method="saliency",
        attributed_fn="probability",
        context_sensitivity_metric="kl_divergence",
        context_sensitivity_std_threshold=2,
        attribution_std_threshold=2,
        attribution_topk=5,
        input_current_text=sentence_eng_Latn,
        input_context_text=context_eng_Latn,
        contextless_input_current_text="""{current}""",
        input_template="""{context} {current}""",
        output_context_text=context_target_language,
        contextless_output_current_text="""{current}""",
        output_template="{context} {current}",
        save_path=sub_directory_name + ".json",
        viz_path=sub_directory_name + ".html",
        tokenizer_kwargs={'src_lang': 'eng_Latn', 'tgt_lang': language_code},
    )
    
    # Run the PECORE model
    out = attribute_context_with_model(pecore_args, inseq_model)
    
    return out