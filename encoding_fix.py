import json

input_file_path = 'extracted_relevant_data_with_context.json'
output_file_path = 'fixed_encoding.json'

data_to_save = []

with open(input_file_path, 'r', encoding='utf-8') as input_file:
    for line in input_file:
        try:
            data = json.loads(line)
            data_to_save.append(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(data_to_save, output_file, ensure_ascii=False, indent=4)

