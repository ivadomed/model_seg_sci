"""
Get filenames with the ✅ or ⚠️ symbols from the input JSON file. These symbols are hardcoded below --> modify the
code if you need to match different symbols.
The JSON file is obtained from SCT's QC html report by downloading it using the "Save All" button.

Example usage:
    python extract_passed_files.py -i qc_report.json -o qc_pass.yml

Author: Jan Valosek
"""

import json
import re
import sys
import argparse
import yaml


def extract_successful_filenames(json_str):
    # Parse JSON string
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []

    # List to store extracted filenames
    successful_files = []

    # Pattern to match the filename portion (sub-*_acq-sag*_T2w.nii.gz)
    pattern = r'_?(sub-[^_]+_acq-sag(?:_run-\d+)?_T2w\.nii\.gz)'

    # Iterate through the JSON entries
    for key, value in data.items():
        if value == "✅" or value == "⚠️":
            # Try to extract filename using regex
            match = re.search(pattern, key)
            if match:
                filename = match.group(1)
                successful_files.append(filename)

    return successful_files


def save_to_yaml(filenames, output_file):
    # Create a dictionary with the list of filenames
    data = {'T2w_sag': filenames}

    # Save to YAML file
    try:
        with open(output_file, 'w') as file:
            yaml.safe_dump(data, file, default_flow_style=False, sort_keys=False)
        print(f"Successfully saved {len(filenames)} filenames to {output_file}")
    except Exception as e:
        print(f"Error saving YAML file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Extract successful filenames from JSON and save to YAML')
    parser.add_argument('-i', help='Path to the input JSON file', required=True)
    parser.add_argument('-o', default='qc_pass_artifact.yml',
                        help='Path to the output YAML file (default: qc_pass.yml)', required=False)
    args = parser.parse_args()

    try:
        # Read and process JSON file
        with open(args.i, 'r') as file:
            json_str = file.read()
            successful_files = extract_successful_filenames(json_str)

            # Print to console
            print(f"Found {len(successful_files)} successful files:")
            for filename in successful_files:
                print(filename)

            # Save to YAML
            save_to_yaml(successful_files, args.o)

    except FileNotFoundError:
        print(f"Error: File '{args.i}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()