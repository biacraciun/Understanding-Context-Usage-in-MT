from setup_data_pecore import setup_pecore, use_pecore
from use_pecore import use_pecore, load_model

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
    # Load the model speecific to the target language
    model = load_model(langugage_code)

    # Indices of the examples to run the PECORE model on
    indices = [12, 18, 25, 29, 39, 53]
    
    # Run the PECORE model for the target language 
    for i in indices:
        use_pecore(data["sentence_eng_Latn"][i], data["sentence_context_eng_Latn"][i], data["sentence_context_" + langugage_code][i], langugage_code, model)
        
    
if __name__ == "__main__":
    
    #langugage_codes = ['bul_Cyrl', 'nld_Latn', 'ron_Latn']
    
    run_pecore('ron_Latn')
    
    