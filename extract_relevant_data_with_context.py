from datasets import load_dataset
import spacy
from spacy.cli.download import download
import pandas as pd

# Variants of pronouns you, your, yours in different languages adapted from MuDA repository
# https://github.com/CoderPat/MuDA

variants_romanian = [
    "tu", "tine", "te", "te-", "îți", "ți-", "voi", "vouă", "vă", "v-", "vi", "vi-", "tău", "ta", "tale",
    "tăi", "vostru", "voastra", "voastre", "vostri",
    "dumneavoastră", "dumneata", "mata", "matale", "dânsul", "dânsa", "dumnealui", "dumneaei", "dumnealor",
    "dvs.", "d-voastră", "dv.", "dumneasa", "d-sa", "dumisale", "d-sale", "d-lui", "d-ei", "d-lor", "d-ta",
    "dumitale", "d-tale"
]

variants_english = ["you", "your", "yours"]

variants_dutch = ["jij", "jouw", "jou", "jullie", "je", "u", "men", "uw"]

pronouns_bulgarian = [
    "Вие", "ти", "Вашият", "Вашия", "твоят", "твоя", "вашата",
    "ваша", "твоята", "твоя","вашите", "ваши", "твоите", "твои"
]

def check_pronouns(sent, language, nlp):
    """Check if a sentence contains the pronouns you, your, yours.
    Args:
        sent (str): The sentence to check.
        language (str): The language of the sentence.
        nlp (spacy.lang): The spacy language model for the language.
    Returns:
        bool: True if the sentence contains the pronouns you, your, yours, False otherwise.
    """

    pronouns_variants = []

    if language == 'ro':
        pronouns_variants = variants_romanian

    elif language == 'nl':
        pronouns_variants = variants_dutch

    elif language == 'en':
        pronouns_variants = variants_english

    # elif language == 'bulgarian':
    #     pronouns_variants = variants_bulgarian

    # use spacy to tokenize the sentence, transform to lowercase and store in a set
    doc = nlp(sent)
    words = set()
    for token in doc:
        words.add(token.text.lower())

    # check if any of the pronouns are in the set of words
    for pronoun in pronouns_variants:
        if pronoun in words:
            return True
    return False

def extract_context_starting_index(i, split, context_window=4):
    """Extract the starting index of the context window for a given sentence.
    Args:
        i (int): The index of the sentence.
        split (str): The split of the dataset.
        context_window (int): The maximum size of the context window (default 4).
    Returns:
        int: The starting index of the context window.
    """

    maximum_context_window = context_window
    context_starting_index = 0

    while True:
        # check if the context window is within the bounds of the dataset
        if i - maximum_context_window >= 0:
            # if the URL is the same, set the context starting index
            if datasets[default_language][split]["URL"][i] == datasets[default_language][split]["URL"][i - maximum_context_window]:
                context_starting_index = i - maximum_context_window
                break
            # if the URL is different, decrease the context window size
            else:
                maximum_context_window -= 1
        # if the context window is too large, decrease the context window size
        else:
            maximum_context_window -= 1
        # if the context window size is 0, return None
        if context_starting_index - i == 0:
            return None
    return context_starting_index

def load_spacy_models(model_name):
    """ Load spacy models and download them if missing.
    Args:
        model_name (str): The name of the spacy model to load.
    Returns:
        spacy.lang: The spacy language model.
    """

    try:
        return spacy.load(model_name)
    except:
        try:
            download(model_name)
            return spacy.load(model_name)
        except:
            return None
        
        
def setup_spacy_models(language_codes):
    """
    Load spacy models for the given language codes.
    Args:
        language_codes (list): The list of language codes.
    Returns:
        dict: A dictionary of spacy language models.
    """
    
    nlps = dict()
    
    for language_code in language_codes:
        if language_code == 'en':
            nlps[language_code] = load_spacy_models('en_core_web_sm')
        else:
            nlps[language_code] = load_spacy_models(language_code + '_core_news_sm')
    
    return nlps
    

def load_datasets(languages):
    """ Load the FLORES-200 dataset for the given languages.
    Args:
        languages (list): The list of languages.
    Returns:
        dict: A dictionary of datasets.
    """

    datasets = dict()

    for language in languages:
        datasets[language] = load_dataset("facebook/flores", language)
    
    return datasets

def extract_data_with_context(datasets, splits, codes_flores, default_language, nlps):
    """ Extract the relevant data with context for each language. 
    Args:
        datasets (dict): The dictionary of datasets.
        splits (list): The list of splits.
        codes_flores (list): The list of language codes.
        default_language (str): The default language code.
        nlps (dict): The dictionary of spacy language models.
    Returns:
        pd.DataFrame: The extracted data with context for each language.
    """
    
    final_dataset = pd.DataFrame()

    # iterate over the splits
    for split in splits:
        
        indices_by_split = []
        context_by_split = dict()
        dataset_by_split = pd.DataFrame()

        for language in codes_flores:
            context_by_split[language] = []

        # iterate over the sentences from the default language due
        # to the subject not being always expressed in Romanian sentences
        # So, it is the most selective language to filter the data
        for i, sentence in enumerate(datasets[default_language][split]["sentence"]):
            
            # check if the sentence contains the pronouns you, your, yours in all languages
            if check_pronouns(sentence, default_language.split('_')[0][:2], nlps[default_language.split('_')[0][:2]])\
            and not any(not check_pronouns(datasets[language][split]["sentence"][i], language.split('_')[0][:2],\
            nlps[language.split('_')[0][:2]])\
            for language in codes_flores if language != default_language and language != 'bul_Cyrl'):
                            
                # store the index of the sentence and extract the context
                indices_by_split.append(i)
                context_indices = extract_context_starting_index(i, split)

                # store the context for each language
                for language in codes_flores:
                    context_by_split[language].append(' '.join(datasets[language][split]["sentence"][context_indices:i]))

        # extract the data by split and indices
        dataset_by_split = datasets[default_language][split][indices_by_split]

        # create the columns for the sentences and the context for each language
        for language in codes_flores:
            dataset_by_split["sentence_" + language] = [datasets[language][split]["sentence"][i] for i in indices_by_split]
            dataset_by_split["sentence_context_" + language] = context_by_split[language]

        # concatenate the data for each split
        final_dataset = pd.concat([final_dataset, pd.DataFrame(dataset_by_split)], ignore_index=True)
        
        # remove the sentence column as it is already stored 
        final_dataset.drop('sentence', axis=1, inplace=True)

    return final_dataset

if __name__ == "__main__":

    codes_flores = ['bul_Cyrl', 'nld_Latn', 'eng_Latn', 'ron_Latn']
    default_language = 'ron_Latn'
    splits = ['dev', 'devtest']
    codes_spacy = ['ro', 'en', 'nl']

    # Load the FLORES-200 dataset for the given languages
    datasets = load_datasets(codes_flores)

    # Load spacy models for the given language codes
    nlps = setup_spacy_models(codes_spacy)

    # Extract the relevant data with context
    final_dataset = extract_data_with_context(datasets, splits, codes_flores, default_language, nlps)
    
    # Save the extracted data with context to a JSON file
    json_extracted_relevant_data_with_context = final_dataset.to_json("data/data_with_context.json", orient='records', lines=True)