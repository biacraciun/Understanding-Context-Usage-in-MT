# Contextual Sensitivity in Machine Translation: A Cross-Linguistic Study of Dutch, Bulgarian, and Romanian

## Outline

### Description

Abstract

### Data

The dataset that is used for investigating how formality transfers given the English source pronouns 'you', 'your', and 'yours' into the three target languages is [FLORES-200](https://huggingface.co/datasets/facebook/flores) dataset. It consists of parallel sentence translations for 200 languages, including Dutch, Romanian, and Bulgarian. 

### Preprocessing and postprocessing 

However, to utilise this dataset, preprocessing is required. First, we need to extract the sentences that contain formality-sensitive pronouns ('you', 'your', and 'yours'), and then we perform the extraction of the preceding context of this sentence. It is important to mention that this context extraction is possible because sentences with consecutive 'IDs' sharing the same 'URL' are consecutive in the text. 

These preprocessing and postprocessing steps are included in the [extract_relevant_data_with_context.py](modules/extract_relevant_data_with_context.py) and, respectively, [setup_data_pecore.py](modules/setup_data_pecore.py) file. 

To identify formality-sensitive pronouns, we drew inspiration from the [Multilingual Discourse-Aware Benchmark (MuDA)](https://github.com/CoderPat/MuDA) repository, but we developed our own language-specific taggers to detect formality. For the second step, we utilize the 'URL' feature to detect context, setting the maximum size of the context window to be 4, equivalent to 4 preceding sentences. The resulting dataset is located at [data_with_context.json](data/data_with_context.json). Before delving into the PECoRe part, we need to filter the data to include only sentences that do not have an empty context. Our final dataset can be found at [filtered_data_with_context.json](data/filtered_data_with_context.json).

### PECoRe 

[PECoRe](https://huggingface.co/spaces/gsarti/pecore), or Plausibility Evaluation of Context Reliance, is an interpretability framework that aims to evaluate and quantify how language models utilize contextual information when generating texts. Using this framework, we can identify the specific tokens that contribute to the translation performed by the model. The exact parameters used for this study can be observed in the [use_pecore.py](modules/use_pecore.py) file.  

## Running the code

### Installation

For installing all dependencies, run the following command:
```bash
pip install -r requirements.txt
```
### Preprocessing and postprocessing data 

The preprocessing part can be performed running the following command:
```bash
python modules/extract_relevant_data_with_context.py
```

The postprocessing step can be achieved by running the following command:
```bash
python modules/setup_data_pecore.py
```

For convenience, we already provided the required datasets for this project in the [data](data) folder. Moreover, the [data_with_context_fixed_encoding.json](data/data_with_context_fixed_encoding.json) is provided for visual inspection of the preprocessed data with the correct encoding adapted to the Bulgarian Cyrillic alphabet and Romanian diacritics. Lastly, this folder also includes the [filtered dataset](data/filtered_data_with_context) exported as a 🤗 dataset such that it can be easily accessible by the PECoRe framework, located in the [modules](modules) folder. The default language is set to Romanian, but it can be changed with ease in the [run_pecore.py](modules/run_pecore.py) file, namely in the main block.

To deploy the PECoRe framework, run the following command: 
```bash
python modules/run_pecore.py
```

### Visualization of the PECoRe output
A notebook for each language is provided for inspecting the generated output of the PECoRe method. These can be found in the [notebooks](notebooks) folder.