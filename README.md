# Contextual Sensitivity in Machine Translation: A Cross-Linguistic Study of Dutch, Bulgarian, and Romanian

## Outline

### Description

Abstract

### Data

The dataset that is used for investigating how formality transfers given the English source pronouns 'you', 'your', and 'yours' into the three target languages is [FLORES-200](https://huggingface.co/datasets/facebook/flores) dataset. It consists of parallel sentence translations for 200 languages, including Dutch, Romanian, and Bulgarian. 

### Pre-processing and post-processing 

However, to utilise this dataset, preprocessing is required. First, we need to extract the sentences that contain formality-sensitive pronouns ('you', 'your', and 'yours'), and then we perform the extraction of the preceding context of this sentence. It is important to mention that this context extraction is possible because sentences with consecutive IDs sharing the same 'URL' are consecutive in the text. 

These preprocessing steps are included in the. 

To identify formality-sensitive pronouns, we drew inspiration from the Multilingual Discourse-Aware Benchmark (MuDA) repository, but we developed our own language-specific taggers to detect formality. For the second step, we utilize the 'URL' feature to detect context, setting the maximum size of the context window to be 4, equivalent to 4 preceding sentences. The resulting dataset can be observed in [...]. Before delving into the PECoRe  part, we need to filter the data to include only sentences that do not have an empty context. Our final dataset can be found in [...]

### PECoRe 



## Running the code

### Installation

For installing all dependencies, run the following command:
```bash
pip install -r requirements.txt
```
### Preprocessing and postprocessing data 

The preprocessing part can be performed running the following command:
```bash
pip extract_relevant_data_with_context.py
```

The postprocessing step can be achieved by running the following command:
```bash
pip run_pecore.py
```

For convinience, we provided both datasets in ... Moreover, the dataset is provided for visual inspection of the preprocessed data with the correct encoding adapted to the Bulgarian Cyrillic alphabet and Romanian diacritics.


### Visualization of the PECoRe output
A notebook for each language is provided for inspecting the generated explanations with inference. These can be found in the [...]() folder.