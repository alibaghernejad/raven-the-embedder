
import json
import os
import flattenformater

def load_dataset_from_dir(dir_path):
    """
    Load a dataset from a directory of JSON files.
    """
    dataset = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".json"):
            file_path = os.path.join(dir_path, filename)
            file_entry = load_json_file(file_path)
            # Exclude trailers by certain fields with title contains "تریلر"
            if file_entry and "title" in file_entry["_source"]:
                if "تریلر" in file_entry["_source"]["title"]:
                    return None
            formatted_entry = format_json_item(file_entry)
            dataset.append(formatted_entry)
    return dataset

def load_json_file(file_path):
    """
    Load a individual file from a JSON file.
    """
    with open(file_path, "r") as f:
        individual_file = json.load(f)
    if isinstance(individual_file, list):
        individual_file = {str(i): v for i, v in enumerate(individual_file)}
    return individual_file

def format_json_item(file_entry):
    """
    Process a single JSON item.
    """
    file_entry["title"] = file_entry["_source"]["title"] if file_entry else ""
    file_entry["text"] = flattenformater.flatten_json_gapfilm([file_entry]) if file_entry else ""
    file_entry["text_en"] = file_entry["_source"]["englishbody"] if file_entry else ""
    return file_entry