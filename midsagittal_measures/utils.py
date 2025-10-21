import pandas as pd

CLINICAL_SCORES_MAX = {
    'uems': 50,  # Maximum UEMS score
    'lems': 50,  # Maximum LEMS score
    'ms': 100,    # Maximum Total Motor Score
    'pp': 112,   # Maximum Pinprick Score
    'lt': 112    # Maximum Light-Touch Score
}

METHOD_TO_TITLE = {
    'GT': 'Semi-automatic (manual lesion masks + SCT)',
    'SCIsegV2': 'Automatic (SCIsegV2 + SCT)'
}

CLINICAL_SCORES_TO_AXES = {
    'uems': 'Upper Extremity Motor Score',
    'lems': 'Lower Extremity Motor Score',
    'ms': 'Total Motor Score',
    'pp': 'Pinprick Score',
    'lt': 'Light-Touch Score'
}

def read_file_manual_sci_zurich(file):
    """
    Read the XLSX file with manually measured metrics and clinical scores for sci-zurich dataset.
    :param file: str: path to the XLSX file
    :return df_manual: pandas DataFrame: dataframe with manually measured lesion metrics and clinical scores
    """
    df_manual = pd.read_excel(file)
    # Drop rows where 'participant_id' is NaN or 'exclude' -- rows with comments
    df_manual = df_manual.dropna(subset=['participant_id'])
    df_manual = df_manual[df_manual['participant_id'] != 'exclude']
    # If session_id is nan in the manual file, set it to 'ses-01'
    df_manual['session_id'] = df_manual['session_id'].fillna('ses-01')

    # Sum up ventral and dorsal tissue bridges to get total tissue bridge
    df_manual['total_tissue_bridge'] = df_manual['ventral_tissue_bridge'] + df_manual['dorsal_tissue_bridge']
    # Rename columns to distinguish manual metrics from SCT metrics
    df_manual.rename(columns={'midsagittal_length': 'midsagittal_length_manual',
                              'midsagittal_width': 'midsagittal_width_manual',
                              'ventral_tissue_bridge': 'ventral_tissue_bridge_manual',
                              'dorsal_tissue_bridge': 'dorsal_tissue_bridge_manual',
                              'total_tissue_bridge': 'total_tissue_bridge_manual'},
                     inplace=True)

    # Compute tissue bridge ratios
    df_manual['dorsal_bridge_ratio_manual'] = df_manual.apply(
        lambda row: (row['dorsal_tissue_bridge_manual'] / row['total_tissue_bridge_manual'] * 100)
        if row['total_tissue_bridge_manual'] > 0 else 0, axis=1)
    df_manual['ventral_bridge_ratio_manual'] = df_manual.apply(
        lambda row: (row['ventral_tissue_bridge_manual'] / row['total_tissue_bridge_manual'] * 100)
        if row['total_tissue_bridge_manual'] > 0 else 0, axis=1)

    # Remove any strings from the 'mri_time_since_injury' column
    if 'mri_time_since_injury' in df_manual.columns:
        df_manual['mri_time_since_injury'] = df_manual['mri_time_since_injury'].astype(str).str.extract(r'(\d+)').astype(int)

    # Convert any string clinical score columns to numeric
    for col in df_manual.columns:
        if col.startswith(tuple(CLINICAL_SCORES_TO_AXES.keys())):
            df_manual[col] = pd.to_numeric(df_manual[col], errors='coerce')

    # Drop 'comment' column and unnamed columns
    df_manual = df_manual.drop(columns=['comment'], errors='ignore')
    unnamed_cols = [col for col in df_manual.columns if 'Unnamed' in col]
    df_manual = df_manual.drop(columns=unnamed_cols, errors='ignore')

    # Reorder the columns to have participant_id, session_id, and mri_time_since_injury first, followed by manual
    # metrics (_manual suffix)
    cols = ['participant_id', 'session_id', 'mri_time_since_injury'] + \
           [col for col in df_manual.columns if col.endswith('_manual')] + \
           [col for col in df_manual.columns if not col.endswith('_manual') and col not in ['participant_id', 'session_id', 'mri_time_since_injury']]
    df_manual = df_manual[cols]

    print(f'Read {len(df_manual)} rows from the manual metrics file: {file}')
    return df_manual


def read_file_manual_nisci_trial(file):
    """
    Read the XLSX file with manually measured metrics and clinical scores for nisci-trial dataset.
    :param file: str: path to the XLSX file
    :return df_manual: pandas DataFrame: dataframe with manually measured lesion metrics and clinical scores
    """
    df_manual = pd.read_excel(file)
    # Renate 'Patient' column to 'participant_id'
    df_manual.rename(columns={'Patient': 'participant_id'}, inplace=True)
    # Rename rows in 'participant_id' from 'sub-zh001' to 'sub-001'
    df_manual['participant_id'] = df_manual['participant_id'].str.replace('sub-zh', 'sub-')

    # Rename columns to distinguish manual metrics from SCT metrics
    df_manual.rename(columns={'lesion_length': 'midsagittal_length_manual',
                              'lesion_width': 'midsagittal_width_manual',
                              'ventral_bridges': 'ventral_tissue_bridge_manual',
                              'dorsal_bridges': 'dorsal_tissue_bridge_manual',
                              'total_bridges': 'total_tissue_bridge_manual'},
                     inplace=True)

    # Rename clinal columns, e.g., 'UEMS_01' to 'uems_01', 'UEMS_02' to 'uems_02', etc.
    for col in df_manual.columns:
        if 'UEMS' in col:
            df_manual.rename(columns={col: col.lower().replace('uems', 'uems')}, inplace=True)
        elif 'LEMS' in col:
            df_manual.rename(columns={col: col.lower().replace('lems', 'lems')}, inplace=True)
        elif 'TMS' in col:
            df_manual.rename(columns={col: col.lower().replace('tms', 'ms')}, inplace=True)
        elif 'TPP' in col:
            df_manual.rename(columns={col: col.lower().replace('tpp', 'pp')}, inplace=True)
        elif 'TLT' in col:
            df_manual.rename(columns={col: col.lower().replace('tlt', 'lt')}, inplace=True)

    # Compute tissue bridge ratios
    df_manual['dorsal_bridge_ratio_manual'] = df_manual.apply(
        lambda row: (row['dorsal_tissue_bridge_manual'] / row['total_tissue_bridge_manual'] * 100)
        if row['total_tissue_bridge_manual'] > 0 else 0, axis=1)
    df_manual['ventral_bridge_ratio_manual'] = df_manual.apply(
        lambda row: (row['ventral_tissue_bridge_manual'] / row['total_tissue_bridge_manual'] * 100)
        if row['total_tissue_bridge_manual'] > 0 else 0, axis=1)

    # Calculate 'mri_time_since_injury' from 'DOI' and 'MRI_time' columns
    df_manual['mri_time_since_injury'] = (df_manual['MRI_time'] - df_manual['DOI']).dt.days

    # Reorder the columns to have participant_id and mri_time_since_injury first, followed by manual
    # metrics (_manual suffix)
    cols = ['participant_id', 'mri_time_since_injury'] + \
           [col for col in df_manual.columns if col.endswith('_manual')] + \
           [col for col in df_manual.columns if not col.endswith('_manual') and col not in ['participant_id', 'mri_time_since_injury']]
    df_manual = df_manual[cols]

    print(f'Read {len(df_manual)} rows from the manual metrics file: {file}')
    return df_manual


def read_file_sct(file_sct):
    """
    Read CSV file with SCT-computed lesion metrics.
    """
    df_sct = pd.read_csv(file_sct)
    # Rename columns to match the manual metrics
    df_sct.rename(columns={'length_interpolated_midsagittal_slice': 'midsagittal_length',
                           'width_interpolated_midsagittal_slice': 'midsagittal_width',
                           'interpolated_dorsal_bridge_width': 'dorsal_tissue_bridge',
                           'interpolated_ventral_bridge_width': 'ventral_tissue_bridge',
                           'interpolated_total_bridge_width': 'total_tissue_bridge'},
                  inplace=True)
    # Add suffix to all columns except participant_id and session_id
    df_sct = df_sct.add_suffix('_sct')
    df_sct.rename(columns={'participant_id_sct': 'participant_id', 'session_id_sct': 'session_id'}, inplace=True)

    print(f'Read {len(df_sct)} rows from the SCT metrics file: {file_sct}')
    return df_sct


def read_participants_file(file):
    """
    Read the participants.tsv file with demographic information.
    :param file: str: path to the TSV file
    :return df_participants: pandas DataFrame: dataframe with demographic information
    """
    df_participants = pd.read_csv(file, sep='\t')

    # Convert columns to appropriate types
    if 'age' in df_participants.columns:
        df_participants['age'] = pd.to_numeric(df_participants['age'], errors='coerce')

    if 'MagneticFieldStrength' in df_participants.columns:
        df_participants['MagneticFieldStrength'] = pd.to_numeric(df_participants['MagneticFieldStrength'], errors='coerce')

    print(f'Read {len(df_participants)} participants from the demographics file: {file}')
    return df_participants


def format_pvalue(p_value, alpha=0.05):
    """
    Format p-value for display.
    """
    if p_value < 0.001:
        return 'p < 0.001'
    elif p_value < alpha:
        return f'p < {alpha}'
    else:
        return f'p = {p_value:.3f}'


def normalize_sensorimotor_scores(df, time_points):
    """
    Compute normalized recovery rates by calculating the change from baseline (or the first available measurement) to
    follow-up and dividing them by the maximal score improvable.
        Refs:
            - http://dx.doi.org/10.1016/S1474-4422(24)00173-X
            - http://dx.doi.org/10.1016/j.nicl.2023.103339
    Then, scale the normalized sensorimotor recovery rates using min-max scaling method (per participant and score):
        x' = (x - min(x)) / (max(x) - min(x))

    For each subject:
        1. Identifies the first available time point (baseline or 1m)
        2. Computes maximal improvable score from this first time point
        3. Normalizes subsequent scores by dividing improvement by maximal improvable score
        4. Applies min-max scaling to the normalized recovery rates per participant and score

    :param df: pandas DataFrame with baseline lesion metrics and clinical scores across
    multiple time points
    : time_points list: list of time points in order (e.g., ['bl', '1m', '3m', '6m', '12m'] or
    ['01', '02', '03', '04', '05', '06'])
    :return df: pandas DataFrame with min-max scaled normalized clinical scores
    """
    # Get unique participant IDs
    participants = df['participant_id'].unique()

    # First pass: compute normalized recovery rates by calculating the change from baseline (i.e., first available
    # measurement) to follow-up and dividing them by the maximal score improvable
    for participant in participants:
        # Get data for the current participant
        participant_data = df[df['participant_id'] == participant]

        # Process each clinical score
        for score in CLINICAL_SCORES_TO_AXES.keys():
            # Find first available time point for this score
            first_tp = None
            first_value = None

            # Check time points in order (baseline first, then 1m)
            for tp in time_points:
                col_name = f"{score}_{tp}"
                if col_name in participant_data.columns and not pd.isna(participant_data[col_name].values[0]):
                    first_tp = tp
                    first_value = participant_data[col_name].values[0]
                    break

            # Skip if no data available for this score
            if first_tp is None:
                continue

            # Get maximum possible score for this clinical measure
            max_score = CLINICAL_SCORES_MAX[score]

            # Calculate maximal improvable score (difference between max possible and first value)
            max_improvable = max_score - first_value

            # Find all subsequent time points to normalize
            subsequent_tps = time_points[time_points.index(first_tp) + 1:]

            # Normalize each follow-up time point after the first available one
            for tp in subsequent_tps:
                follow_up_col = f"{score}_{tp}"
                if follow_up_col in participant_data.columns and not pd.isna(participant_data[follow_up_col].values[0]):
                    # Calculate improvement from first time point
                    follow_up_value = participant_data[follow_up_col].values[0]
                    improvement = follow_up_value - first_value

                    # Create new column for normalized score
                    normalized_col = f"{follow_up_col}_improvement_normalized"

                    # Normalize improvement by maximal improvable score
                    if max_improvable <= 0:  # If no room for improvement, set to 0
                        df.loc[participant_data.index, normalized_col] = 0  # or should I use `np.nan`?
                    else:
                        normalized_improvement = improvement / max_improvable
                        df.loc[participant_data.index, normalized_col] = normalized_improvement

    # Second pass: apply min-max scaling to normalized recovery rates (separately for each participant and
    # clinical score)
    normalized_columns = [col for col in df.columns if col.endswith('_improvement_normalized')]

    for participant in participants:
        participant_idx = df['participant_id'] == participant
        participant_data = df.loc[participant_idx]

        # Process each clinical score separately
        for score in CLINICAL_SCORES_TO_AXES.keys():
            # Get normalized values for this participant and this specific clinical score
            score_normalized_values = []
            score_columns = []

            # Debug - subject with decreased light touch score for 1m from baseline
            # if participant == 'sub-zh12' and score == 'lt':
            #     print('here')

            for col in normalized_columns:
                if col.startswith(f"{score}_") and col in participant_data.columns and not pd.isna(participant_data[col].values[0]):
                    score_normalized_values.append(participant_data[col].values[0])
                    score_columns.append(col)

            # Apply min-max scaling if there are values for this participant and this score
            if len(score_normalized_values) > 0:
                min_val = min(score_normalized_values)
                max_val = max(score_normalized_values)

                # Apply min-max scaling: x' = (x - min(x)) / (max(x) - min(x))
                if max_val != min_val:  # Avoid division by zero
                    for col in score_columns:
                        original_val = participant_data[col].values[0]
                        scaled_val = (original_val - min_val) / (max_val - min_val)
                        # Create new column with _scaled suffix
                        scaled_col = col.replace('_improvement_normalized', '_improvement_normalized_scaled')
                        df.loc[participant_idx, scaled_col] = scaled_val
                else:
                    # If all values are the same for this participant and score, set to 0.5 (middle value)
                    for col in score_columns:
                        # Create new column with _scaled suffix
                        scaled_col = col.replace('_improvement_normalized', '_improvement_normalized_scaled')
                        df.loc[participant_idx, scaled_col] = 0.5

    return df
