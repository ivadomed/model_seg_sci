###
# Prerequirements:
# dcm2bids
# dcm2niix

# How to use ?
# python curate_bids_ucd.py -i PATH_TO_DATASET_TO_CONVERT -o OUTPUT_PATH -c PATH_TO_CONFIG_FILE

import os
import shutil
import json
import argparse


def get_parameters():
    parser = argparse.ArgumentParser(description='Convert dataset to BIDS format.')
    parser.add_argument("-i", "--path-input",
                        help="Path to folder containing the dataset to convert to BIDS",
                        required=True)
    parser.add_argument("-o", "--path-output",
                        help="Path to the output BIDS folder",
                        required=True,
                        )
    parser.add_argument("-c", "--path-config-file",
                        help="Path to the config file for dcm2bids",
                        required=True)
    arguments = parser.parse_args()
    return arguments


def main(path_input, path_output, path_config_file):
    if os.path.isdir(path_output):
        shutil.rmtree(path_output)
    os.makedirs(path_output, exist_ok=True)

    for subject in os.listdir(path_input):
        bids_subject_name = 'sub-' + subject
        path_data_subject = os.path.join(path_input,subject)
        command = 'dcm2bids -d ' + path_data_subject + ' -p ' + bids_subject_name + ' -c ' + path_config_file + ' -o ' + path_output
        os.system(command)
    shutil.rmtree(os.path.join(path_output,'tmp_dcm2bids'))
    sub_list = os.listdir(path_output)
    sub_list.sort()

    import csv

    participants = []
    for subject in sub_list:
        row_sub = []
        row_sub.append(subject)
        row_sub.append('n/a')
        row_sub.append('n/a')
        participants.append(row_sub)

    print(participants)
    with open(path_output + '/participants.tsv', 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(["participant_id", "sex", "age"])
        for item in participants:
            tsv_writer.writerow(item)

    # Create participants.json
    data_json = {"participant_id": {
        "Description": "Unique Participant ID",
        "LongName": "Participant ID"
        },
        "sex": {
            "Description": "M or F",
            "LongName": "Participant sex"
        },
        "age": {
            "Description": "yy",
            "LongName": "Participant age"}
    }

    with open(path_output + '/participants.json', 'w') as json_file:
        json.dump(data_json, json_file, indent=4)

    # Create dataset_description.json
    dataset_description = {"BIDSVersion": "BIDS 1.6.0",
                           "Name": "sci-lesion-segmentation"
                           }

    with open(path_output + '/dataset_description.json', 'w') as json_file:
        json.dump(dataset_description, json_file, indent=4)

    # Create README
    with open(path_output + '/README', 'w') as readme_file:
        readme_file.write('Dataset for sct-pipeline/sci-lesion-segmentation.')


if __name__ == "__main__":
    args = get_parameters()
    main(args.path_input, args.path_output, args.path_config_file)
