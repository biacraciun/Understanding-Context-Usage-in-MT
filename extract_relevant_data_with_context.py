from datasets import load_dataset
import spacy
from spacy.cli.download import download
import pandas as pd

variants_romanian = [
    "tu", "tine", "te", "te-", "îți", "ți-", "voi", "vouă", "vă", "v-", "vi", "vi-", "tău", "ta", "tale",
    "tăi", "vostru", "voastra", "voastre", "vostri",
    "dumneavoastră", "dumneata", "mata", "matale", "dânsul", "dânsa", "dumnealui", "dumneaei", "dumnealor",
    "dvs.", "d-voastră", "dv.", "dumneasa", "d-sa", "dumisale", "d-sale", "d-lui", "d-ei", "d-lor", "d-ta",
    "dumitale", "d-tale"
]

variants_english = ["you", "your", "yours"]

variants_dutch = ["jij", "jouw", "jou", "jullie", "je", "u", "men", "uw"]

pronouns_bulgarian = ["Вие", "ти", "Вашият", "Вашия", "твоят", "твоя", "вашата", "ваша", "твоята", "твоя", "вашите", "ваши", "твоите", "твои"]

def check_formality(sent, language, nlp):

    spacy_label = ""
    pronouns_variants = []

    if language == 'ro':
        pronouns_variants = variants_romanian

    elif language == 'nl':
        pronouns_variants = variants_dutch

    elif language == 'en':
        pronouns_variants = variants_english

    # elif language == 'bulgarian':
    #     pronouns_variants = variants_bulgarian

    doc = nlp(sent)
    words = set()
    for token in doc:
        words.add(token.text.lower())

    for pronoun in pronouns_variants:
        if pronoun in words:
            return True
    return False

def extract_context_starting_index(i, split, context_window=4):

    maximum_context_window = context_window
    context_starting_index = 0

    while True:
        if i - maximum_context_window >= 0:
            if datasets[default_language_code_flores][split]["URL"][i] == datasets[default_language_code_flores][split]["URL"][i - maximum_context_window]:
                context_starting_index = i - maximum_context_window
                break
            else:
                maximum_context_window -= 1
        else:
            maximum_context_window -= 1
        if context_starting_index - i == 0:
            return None
    return context_starting_index

def load_spacy_models(model_name):
    """ """

    try:
        return spacy.load(model_name)
    except:
        try:
            download(model_name)
            return spacy.load(model_name)
        except:
            return None

def load_datasets(languages):

    datasets = dict()

    for language in languages:
        datasets[language] = load_dataset("facebook/flores", language)
    
    return datasets

def extract_data_with_context(datasets, splits, languages_codes_flores, default_language_code_flores, nlps):
    final_dataset = pd.DataFrame()

    for split in splits:

        indices_by_split = []
        context_temp = []
        context_by_split = dict()
        dataset_by_split = pd.DataFrame()

        for language in languages_codes_flores:
            context_by_split[language] = []

        for i, sentence in enumerate(datasets[default_language_code_flores][split]["sentence"]):

            if check_formality(sentence, 'ro', nlps['ro']) and\
            check_formality(datasets["eng_Latn"][split]["sentence"][i], 'en', nlps['en']) and\
            check_formality(datasets["nld_Latn"][split]["sentence"][i], 'nl', nlps['nl']):

                indices_by_split.append(i)
                context_indices = extract_context_starting_index(i, split)

                for language in languages_codes_flores:
                    context_by_split[language].append(' '.join(datasets[language][split]["sentence"][context_indices:i]))

        dataset_by_split = datasets[default_language_code_flores][split][indices_by_split]

        for language in languages_codes_flores:
            dataset_by_split["sentence_" + language] = [datasets[language][split]["sentence"][i] for i in indices_by_split]
            dataset_by_split["sentence_context_" + language] = context_by_split[language]

        final_dataset = pd.concat([final_dataset, pd.DataFrame(dataset_by_split)], ignore_index=True)
        final_dataset.drop('sentence', axis=1, inplace=True)

    return final_dataset

if __name__ == "__main__":

    languages_codes_flores = ['bul_Cyrl', 'nld_Latn', 'eng_Latn', 'ron_Latn']
    default_language_code_flores = 'ron_Latn'
    splits = ['dev', 'devtest']
    languages_codes_spacy = ['ro', 'en', 'nl']

    datasets = load_datasets(languages_codes_flores)

    nlps = dict()

    # Load models and download them if missing
    nlps['ro'] = load_spacy_models('ro_core_news_sm')
    nlps['en'] = load_spacy_models('en_core_web_sm')
    nlps['nl'] = load_spacy_models('nl_core_news_sm')
    
    # Not sure why this is not working for 'en'

    # for language in languages_codes_spacy:
    #     nlps[language] = load_spacy_models(language + '_core_news_sm')
    # print(nlps['en'])

    final_dataset = extract_data_with_context(datasets, splits, languages_codes_flores, default_language_code_flores, nlps)
    
    json_extracted_relevant_data_with_context = final_dataset.to_json("data/data_with_context.json", orient='records', lines=True)