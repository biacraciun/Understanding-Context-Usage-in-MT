from setup_data_pecore import setup_pecore, use_pecore
from use_pecore import use_pecore, load_model

import os

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
    
    # Create the output directory per language if it does not exist
    directory_path = os.path.join("pecore_output", langugage_code)
    
    if not os.path.exists(directory_path):
       os.makedirs(directory_path)

    # Indices of the examples to run the PECORE model on
    indices = [12, 18, 25, 29, 39, 53]
    
    # Run the PECORE model for the target language 
    for i in indices:
        
        # Create the sub-directory for saving the output 
        sub_directory_name = os.path.join(directory_path, "example_" + str(i))
        
        use_pecore(data["sentence_eng_Latn"][i], data["sentence_context_eng_Latn"][i], data["sentence_context_" + langugage_code][i], langugage_code, model, sub_directory_name)
        
    
if __name__ == "__main__":
    
    #langugage_codes = ['bul_Cyrl', 'nld_Latn', 'ron_Latn']
    
    # Create the output directory for saving the output if it does not exist
    if not os.path.exists("pecore_output"):
       os.makedirs("pecore_output")
    
    run_pecore('ron_Latn')
    

    
    