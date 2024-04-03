import json
from transformers import (
    set_seed
)
import pandas as pd
from datasets import Dataset
import os
from use_pecore import use_pecore

def open_data(input_file_path='data/data_with_context.json'):
    """
    This function opens the data set with the correct encoding, and returns it.\
    If the file does not exist, it creates it.
    Args:
        input_file_path (str): The path to the data set.
    Returns:
        list: The data set as a list of dictionaries.
    """
    
    output_file_path = 'data/data_with_context_fixed_encoding.json'

    data_to_save = []

    # Open the data set
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            try:
                data = json.loads(line)
                data_to_save.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    
    # Save the data set with the correct encoding if it does not exist
    if not os.path.exists(output_file_path):
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(data_to_save, output_file, ensure_ascii=False, indent=4)
        
    return data_to_save

    
def setup_pecore():
    """
    This function sets up the PECORE model by opening the data set,\
    filtering it (selecting only examples with context).
    Args:
        None
    Returns:
        Dataset: The filtered data set.
    """

    # Convert to Pandas DataFrame to filter data
    df = pd.DataFrame(open_data())
    
    # Only select examples with context
    filtered_df = df.loc[(df["sentence_context_eng_Latn"].apply(lambda x: len(x) > 0))]

    # Convert back to Dataset object
    data = Dataset.from_pandas(filtered_df)

    # Set seed for reproducibility
    random_seed = 0
    set_seed(random_seed)

    # Randomly select 10 examples
    #data_10_exmples = data.shuffle(seed=random_seed)[:10]
    
    data.to_json("data/filtered_data_with_context.json", orient='records', lines=True)
    data.save_to_disk("data/filtered_data_with_context")

    return data
    
if __name__ == "__main__":
    
    data = setup_pecore()