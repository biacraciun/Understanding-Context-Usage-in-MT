from datasets import load_dataset
import spacy
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

    if language == 'romanian':
        pronouns_variants = variants_romanian

    elif language == 'dutch':
        pronouns_variants = variants_dutch

    elif language == 'english':
        pronouns_variants = variants_english

    # elif language == 'bulgarian':
    #     pronouns_variants = variants_bulgarian
    
    doc = nlp(sent)
    words = []
    for token in doc:
        words.append(token.text.lower())

    for pronoun in pronouns_variants:
        if pronoun in words:
          return True
    return False
    
def extract_context(url, i, split, context_window=4):

  maximum_context_window = context_window
  context = []

  while True:
    if i - maximum_context_window >= 0:
      if dataset_romanian[split]["URL"][i] == dataset_romanian[split]["URL"][i - maximum_context_window]:
        context = dataset_romanian[split]["sentence_eng_Latn"][i - maximum_context_window:i]
        break
      else:
        maximum_context_window -= 1
    else:
      maximum_context_window -= 1     
  return context

if __name__ == "__main__":

    dataset_romanian = load_dataset("facebook/flores", "eng_Latn-ron_Latn")
    dataset_dutch = load_dataset("facebook/flores", "eng_Latn-nld_Latn")
    dataset_bulgarian = load_dataset("facebook/flores", "eng_Latn-bul_Cyrl")

    splits = ['dev', 'devtest']

    nlp_ro = spacy.load('ro_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')
    nlp_nl = spacy.load('nl_core_news_sm')  

    final_dataset = pd.DataFrame()
    
    for split in splits:
        indices_by_split = []
        context_by_split = []
        dataset_by_split = pd.DataFrame()

        for i, sentence in enumerate(dataset_romanian[split]["sentence_ron_Latn"]):
            
          if check_formality(sentence, 'romanian', nlp_ro) and\
          check_formality(dataset_romanian[split]["sentence_eng_Latn"][i], 'english', nlp_en) and\
          check_formality(dataset_dutch[split]["sentence_nld_Latn"][i], 'dutch', nlp_nl):
            indices_by_split.append(i)
            context_by_split.append(extract_context(dataset_romanian[split]['URL'][i], i, split))
        
        dataset_by_split = dataset_romanian[split][indices_by_split]
        dataset_by_split["sentence_nld_Latn"] = [dataset_dutch[split]["sentence_nld_Latn"][i] for i in indices_by_split]
        dataset_by_split["sentence_bul_Cyrl"] = [dataset_bulgarian[split]["sentence_bul_Cyrl"][i] for i in indices_by_split]
        dataset_by_split["sentence_context"] = context_by_split
            
        final_dataset = pd.concat([final_dataset, pd.DataFrame(dataset_by_split)], ignore_index=True)