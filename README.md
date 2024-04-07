# Contextual Sensitivity in Machine Translation: A Cross-Linguistic Study of Dutch, Bulgarian, and Romanian

## Outline

### Description

This study explores the influence of the preceding context on the use of formality for the pronouns 'you/your/yours' in machine translation. Different languages often have their structure, and therefore, various context cues contribute to the selection of the pronoun. In this study, we compare three languages from three different language families: Bulgarian (Cyrillic), Romanian (Latin), and Dutch (Germanic). For this purpose, sentences containing the desired pronouns have been extracted from the [FLORES-200](https://huggingface.co/datasets/facebook/flores) dataset, along with their preceding context. The [PECoRe](https://huggingface.co/spaces/gsarti/pecore) framework is used for the analysis, emphasizing contextual details and their impact on translation choices. The results reveal differences among the three languages. Bulgarian and Romanian translations consider verbs and context for pronoun formality, whereas Dutch struggles with formality levels, relying more on specific tokens than overall context.

### Data

The dataset that is used for investigating how formality transfers given the English source pronouns 'you', 'your', and 'yours' into the three target languages is FLORES-200 dataset. It consists of parallel sentence translations for 200 languages, including Dutch, Romanian, and Bulgarian. 

### Preprocessing and postprocessing 

However, to utilise this dataset, preprocessing is required. First, we need to extract the sentences that contain formality-sensitive pronouns ('you', 'your', and 'yours'), and then we perform the extraction of the preceding context of this sentence. It is important to mention that this context extraction is possible because sentences with consecutive 'IDs' sharing the same 'URL' are consecutive in the text. 

These preprocessing and postprocessing steps are included in the [extract_relevant_data_with_context.py](modules/extract_relevant_data_with_context.py) and, respectively, [setup_data_pecore.py](modules/setup_data_pecore.py) file. 

To identify formality-sensitive pronouns, we drew inspiration from the [Multilingual Discourse-Aware Benchmark (MuDA)](https://github.com/CoderPat/MuDA) repository, but we developed our own language-specific taggers to detect formality. For the second step, we utilize the 'URL' feature to detect context, setting the maximum size of the context window to be 4, equivalent to 4 preceding sentences. The resulting dataset is located at [data_with_context.json](data/data_with_context.json). Before delving into the PECoRe part, we need to filter the data to include only sentences that do not have an empty context. Our final dataset can be found at [filtered_data_with_context.json](data/filtered_data_with_context.json).

### PECoRe 

PECoRe or Plausibility Evaluation of Context Reliance, is an interpretability framework that aims to evaluate and quantify how language models utilize contextual information when generating texts. Using this framework, we can identify the specific tokens that contribute to the translation performed by the model. The exact parameters used for this study can be observed in the [use_pecore.py](modules/use_pecore.py) file.  

### DivEMT Dutch
We additionally look into the ability of contextual information to predict translation errors in Dutch: are machine translation errors caused by an unreasonable use of contextual information?

We randomly select 50 Dutch translations from [mBART1-to-50](https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt), sourced from the [DivEMT](https://huggingface.co/datasets/GroNLP/divemt) data set, which we manually analyse through the PECoRe and [Inseq](https://github.com/inseq-team/inseq) architectures. In this analysis, we focus on the tokens that were modified (insertions, deletions, substitutions, and shifts) during post-editing by a professional translator and assess if their invalidity is the result of relying on incorrect context.  

We extract these randomly selected 50 Dutch mBART translations using [get_data.py](get_data.py), which outputs the sentences as a 🤗 Dataset object to the [data/divemt_data](data/divemt_data) folder.
Subsequently, we call [divemt_dutch.py](divemt_dutch.py) to parse the mBART translations accompanied by the requested data (source text, target text andcontext) to both PECoRE and Inseq. Their outputs can be found in [divemt_dutch_analyse](divemt_dutch_analyse), ordered by sentence number. 
Note that PECoRe requires context, sometimes a randomly selected is the first of its paragraph for which is has no (preceding) context. As a result few sentence folders do not contain PECoRe output (sentences: [1](divemt_dutch_analyse/sen_1/), [4](divemt_dutch_analyse/sen_4/) and [9](divemt_dutch_analyse/sen_9/).)


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

For convenience, we already provided the required datasets for this project in the [data](data) folder. Moreover, the [data_with_context_fixed_encoding.json](data/data_with_context_fixed_encoding.json) is provided for visual inspection of the preprocessed data with the correct encoding adapted to the Bulgarian Cyrillic alphabet and Romanian diacritics. Lastly, this folder also includes the [filtered dataset](data/filtered_data_with_context) exported as a 🤗 Dataset object such that it can be easily accessible by the PECoRe framework, located in the [modules](modules) folder. The default language is set to Romanian, but it can be changed with ease in the [run_pecore.py](modules/run_pecore.py) file, namely in the main block.

To deploy the PECoRe framework, run the following command: 
```bash
python modules/run_pecore.py
```

### Visualization of the PECoRe output
A notebook for each language is provided for inspecting the generated output of the PECoRe method. These can be found in the [notebooks](notebooks) folder.

### DivEMT Dutch
The replication of our DivEMT Dutch research direction involves only two scripts:

To retrieve the desired data from DivEMT:
```bash
python get_data.py
```

To extract the context and parse the translations to PECoRe and Inseq:
```bash
python divemt_dutch.py
```