from setup_data_pecore import setup_pecore, use_pecore
from use_pecore import use_pecore, load_model
from datasets import load_from_disk
import os
import sys

def run_pecore(langugage_code, notebook=False):
    """
    This function runs the PECORE model on the data after filtering it.
    Args:
        langugage_code (str): The language code of the target language.
        notebook (bool): Whether the code is running in a notebook environment.
    Returns:
        None
    """
    
    # Load the data if running in a notebook environment
    if notebook:
        data = load_from_disk("Understanding-Context-Usage-in-MT/data/filtered_data_with_context")
    
    else:
        # Set up the data for the PECORE model if it does not exist
        if not os.path.exists("data/filtered_data_with_context"):
            data = setup_pecore()
            
        # Load the data if it exists
        else:
            data = load_from_disk("data/filtered_data_with_context") 
            
    # Load the model speecific to the target language
    model = load_model(langugage_code)
    
    # Create the output directory for saving the output if it does not exist
    if not os.path.exists("notebooks/pecore_output"):
       os.makedirs("notebooks/pecore_output")
    
    # Create the output directory per language if it does not exist
    directory_path = os.path.join("notebooks/pecore_output", langugage_code)
    
    if not os.path.exists(directory_path):
       os.makedirs(directory_path)

    # Indices of the examples to run the PECORE model on

    if langugage_code == 'ron_Latn':
        indices = [12, 18, 25, 29, 39, 53, 37, 44, 1, 2, 16]
    elif langugage_code == 'bul_Cyrl':
        indices = [12, 18, 25, 29, 37, 39, 0, 3, 44, 53, 60]
    elif langugage_code == 'nld_Latn':
        indices = [12, 18, 25, 29, 39, 53, 1, 2, 3, 7, 16]
    
    # Run the PECORE model for the target language 
    for i in indices:
        
        # Create the sub-directory for saving the output 
        sub_directory_name = os.path.join(directory_path, "example_" + str(i))
        
        print(f"Sentence {i}", file=sys.stderr)
        
        use_pecore(data["sentence_eng_Latn"][i], data["sentence_context_eng_Latn"][i], data["sentence_context_" + langugage_code][i], langugage_code, model, sub_directory_name)
        
    
if __name__ == "__main__":
    
    #langugage_codes = ['bul_Cyrl', 'nld_Latn', 'ron_Latn']
    
    run_pecore('ron_Latn', notebook=False)