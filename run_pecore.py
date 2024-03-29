from setup_data_pecore import setup_pecore, use_pecore
from use_pecore import use_pecore

def run_pecore(langugage_code):
    """
    This function runs the PECORE model on the data after setting 
    Args:
        langugage_code (str): The language code of the target language.
    Returns:
        None
    """
    
    # Set up the data for the PECORE model 
    data = setup_pecore()
    # for i in range(len(data)):
    #     use_pecore(data["sentence_eng_Latn"][i], data["sentence_context_eng_Latn"][i], data["sentence_context_" + langugage_code][i], langugage_code)
    
    # Run the PECORE model
    use_pecore(data["sentence_eng_Latn"][0], data["sentence_context_eng_Latn"][0], data["sentence_context_" + langugage_code][0], langugage_code)

if __name__ == "__main__":
    # langugage_codes = ['bul_Cyrl', 'nld_Latn', 'ron_Latn']
    run_pecore(langugage_code='ron_Latn')