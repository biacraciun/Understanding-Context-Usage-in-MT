import json
from transformers import (
    set_seed
)
import pandas as pd
from datasets import Dataset
from use_pecore import use_pecore


def open_data():
    """
    This function opens the data set with the correct encoding, and returns it.
    """

    json_data = []
    with open("extracted_relevant_data_with_context.json", "r", encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                json_data.append(json.loads(line))
    return json_data

    
def setup_pecore():
    """
    This function sets up the PECORE model by opening the data set,\
    filtering it (selecting only examples with context) and\
    randomly selects 10 examples.
    Args:
        None
    Returns:
        Dataset: The filtered data set with 10 randomly examples.
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
    data_10_exmples = data.shuffle(seed=random_seed)[:10]

    return data_10_exmples
