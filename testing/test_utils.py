import os
import re


def fetch_filename_details(filename_path):
    """
    Get dataset name, subject name, session number (if exists), file ID and filename from the input nnUNet-compatible 
    filename or file path. The function works both on absolute file path as well as filename
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz) or file path
    (e.g., /home/user/MRI/bids/derivatives/labels/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
    :return: subjectID: subject ID (e.g., sub-001)
    :return: sessionID: session ID (e.g., ses-01)
    :return: filename: nii filename (e.g., sub-001_ses-01_T1w.nii.gz)
    
    Adapted from: https://github.com/spinalcordtoolbox/manual-correction/blob/main/utils.py#L24
    """

    _, fileName = os.path.split(filename_path)              # Get just the filename (i.e., remove the path)
    datasetName = fileName.split('_')[0]              # Get the dataset name (i.e., remove the filename)
    
    subject = re.search('sub-(.*?)[_/]', filename_path)
    subjectID = subject.group(0)[:-1] if subject else ""    # [:-1] removes the last underscore or slash
    
    session = re.findall(r'ses-..', filename_path)
    sessionID = session[0] if session else ""               # Return None if there is no session

    fID = re.search('_\d{3}', fileName)
    fileID = fID.group(0)[1:] if fID else ""        # [1:-1] removes the underscores

    # REGEX explanation
    # \d - digit
    # \d? - no or one occurrence of digit
    # *? - match the previous element as few times as possible (zero or more times)

    return datasetName, subjectID, sessionID, fileID, fileName



if __name__ == '__main__':
    # test the function
    filename = '/home/GRAMES.POLYMTL.CA/u114716/nnunet-v2/nnUNet_raw/Dataset200_tSCIZurichRPISeed99/imagesTs/tSCIZurichRPISeed99_sub-zh39_ses-01_012_0000.nii.gz'
    # filename = '/home/GRAMES.POLYMTL.CA/u114716/nnunet-v2/nnUNet_raw/Dataset251_tSCIColorado/imagesTs/tSCIColorado_sub-5602_002_0000.nii.gz'
    datasetName, subjectID, sessionID, fileID, fileName = fetch_filename_details(filename)
    print(f'Dataset name: {datasetName}')
    print(f'Subject ID: {subjectID}')
    print(f'Session ID: {sessionID}')
    print(f'File ID: {fileID}')
    print(f'Filename: {fileName}')