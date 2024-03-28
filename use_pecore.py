import inseq
from inseq.commands.attribute_context.attribute_context import (
    AttributeContextArgs,
    attribute_context_with_model,
)

def use_pecore(sentence_eng_Latn, context_eng_Latn, context_target_language, language_code):
    """
    This function uses the PECORE model to attribute the context of a sentence in a target language.
    The code was adapted from the PECORE demo.
    """


    inseq_model = inseq.load_model(
        "facebook/nllb-200-distilled-600M",
        "saliency",
        tokenizer_kwargs={'src_lang': 'eng_Latn', 'tgt_lang': language_code},
    )

    pecore_args = AttributeContextArgs(
        model_name_or_path="facebook/nllb-200-distilled-600M",
        attribution_method="saliency",
        attributed_fn="contrast_prob_diff",
        context_sensitivity_metric="kl_divergence",
        context_sensitivity_std_threshold=0,
        attribution_std_threshold=0,
        attribution_topk=5,
        input_current_text=sentence_eng_Latn,
        input_context_text=context_eng_Latn,
        contextless_input_current_text="""{current}""",
        input_template="""{context} {current}""",
        output_context_text=context_target_language,
        contextless_output_current_text="""{current}""",
        output_template="{context} {current}",
        save_path="pecore_output.json",
        viz_path="pecore_output.html",
        tokenizer_kwargs={'src_lang': 'eng_Latn', 'tgt_lang': language_code},
    )

    out = attribute_context_with_model(pecore_args, inseq_model)
    return out