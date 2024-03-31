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
    model = load_model(langugage_code)
    indices = []
    for i in range(len(data)):
        try:
            output = use_pecore(data["sentence_eng_Latn"][i], data["sentence_context_eng_Latn"][i], data["sentence_context_" + langugage_code][i], langugage_code, model)
            if output.output_current == "":
                continue
            else:
                indices.append(i)
        except:
            continue
    
    return indices
            
    # Run the PECORE model
    
    # use_pecore(data["sentence_eng_Latn"][3], data["sentence_context_eng_Latn"][3], data["sentence_context_" + langugage_code][3], langugage_code, model)
    
if __name__ == "__main__":
    #langugage_codes = ['bul_Cyrl', 'nld_Latn', 'ron_Latn']
    langugage_codes = ['bul_Cyrl', 'nld_Latn', 'ron_Latn']
    indices_list = dict()
    for langugage_code in langugage_codes:
        indices = run_pecore(langugage_code)
        indices_list[langugage_code] = indices
    
    print(indices_list)
    
    # find the intersection of the indices
    indices_intersection = set(indices_list[langugage_codes[0]]).intersection(*indices_list.values())
    print(indices_intersection)
    with open("indices.txt", "w") as f:
        f.write("The indices for the target languages are: " + str(indices_list) + "\n")
        f.write("The intersection of the indices is: " + str(indices_intersection) + "\n")
    
    