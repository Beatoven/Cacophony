import json
import argparse
import os



def merge_jsons(json_files_dir, save_dir):
    # Initialize an empty dictionary to hold all the data

    json_files = [os.path.join(json_files_dir, x) for x in os.listdir(json_files_dir)]
    merged_data = {}

    # Loop through each file
    for file_path in json_files:
        with open(file_path, 'r') as file:
            # Load the current JSON file
            data = json.load(file)
            
            # Loop through each key in the current JSON data
            for key, value in data.items():
                if key in merged_data:
                    # If the key already exists, update/merge the nested dictionary
                    merged_data[key].update(value)
                else:
                    # If the key doesn't exist, add it to the merged dictionary
                    merged_data[key] = value

    # Optionally, save the merged dictionary to a new JSON file
    with open(save_dir, 'w') as file:
        json.dump(merged_data, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts tempo and time of first beat for songs in stem dataset, by summing the stems and using madmom"
    )

    parser.add_argument(
        "-d",
        "--json_files_dir",
        default="embeddings",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        default="cacophony_embeddings.json",
    )
    args = parser.parse_args()
    merge_jsons(args.json_files_dir, args.save_dir)