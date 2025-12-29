"""
Figure 2.

Generate 2x2 figure showing demographic data:
- pie chart for sex
- pie chart for age
- bar chart for AIS grades over time
- bar chart for injury levels

This script:
- reads CSV file with lesion metrics (Note: we don't use the lesion metrics but need participant_id column)
- reads XLSX files with clinical scores for both datasets
- reads TSV files with participant demographics (age, sex, MRI field strength) for both datasets
- merges the data into a single dataframe
- generates a comprehensive figure with multiple subplots (2x2) for descriptive analysis
- creates a table with descriptive statistics

Example usage:
    python 05_descriptive_analysis.py
        -i <PATH_TO_CSV_FILE>
        -file-participants-zurich <PATH_TO_PARTICIPANTS_TSV_ZURICH>
        -file-participants-nisci <PATH_TO_PARTICIPANTS_TSV_NISCI>
        -file-clinical-nisci <PATH_TO_CLINICAL_SCORES_XLSX_NISCI>
        -file-clinical-sci-zurich <PATH_TO_CLINICAL_SCORES_XLSX_SCI_ZURICH>
        -o <OUTPUT_DIR>

Author: Jan Valosek
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings('ignore')

from utils import read_csv_file_with_lesion_metrics, read_participants_file, CLINICAL_SCORES_TO_AXES


CLINICAL_SCORES_MAX = {
    'uems': 50,
    'lems': 50,
    'ms': 100,
    'pp': 112,
    'lt': 112
}

AIS_LABELS = {
    'A': 'Complete',
    'B': 'Sensory Incomplete',
    'C': 'Motor Incomplete',
    'D': 'Motor Incomplete'
}

TETRAPARA_LABELS = {
    0: 'Tetraplegia',
    1: 'Paraplegia'
}

TITLE_SIZE = 20
LABEL_SIZE = FONT_SIZE = TITLE_SIZE - 2
TICK_SIZE = TITLE_SIZE - 6

# Based on cm.Pastel1
PIE_COLORS = ['#FBB4AE', '#B3CDE3', '#CCEBC5', '#DECBE4', '#FED9A6', '#FFFFCC', '#E5D8BD', '#FDDAEC', '#F2F2F2']
TRAJECTORY_COLORS = ['#FBB4AE', '#B3CDE3', '#CCEBC5', '#DECBE4', '#FED9A6']
# AIS A–E: warm → neutral → greenish, same pastel tones as PIE_COLORS
AIS_COLORS = [
    '#FBB4AE',  # A – pastel red/pink
    '#FED9A6',  # B – pastel orange
    '#FFFFCC',  # C – pastel yellow
    '#CCEBC5',  # D – pastel green
    '#B3CDE3',  # E – cool green-blue
]


def get_parser():
    """Parser function for command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate descriptive statistical analysis for SCI lesion data.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        required=True,
        type=str,
        help='Absolute path to a CSV file with lesion metrics with _sct and _manual suffixes.'
    )
    parser.add_argument(
        '-file-participants-zurich',
        required=True,
        type=str,
        help='Absolute path to a TSV file with participant demographics for Zurich (participants.tsv).'
    )
    parser.add_argument(
        '-file-participants-nisci',
        required=True,
        type=str,
        help='Absolute path to a TSV file with participant demographics for NISCI (participants.tsv).'
    )
    parser.add_argument(
        '-file-clinical-nisci',
        required=True,
        type=str,
        help='Absolute path to a XLSX file with participant clinical data for NISCI (clinical_scores.xlsx).'
    )
    parser.add_argument(
        '-file-clinical-sci-zurich',
        required=True,
        type=str,
        help='Absolute path to a XLSX file with participant clinical data for sci-zurich.'
    )
    parser.add_argument(
        '-o',
        required=True,
        type=str,
        help='Path to the output folder where figures and tables will be saved.'
    )

    return parser



def create_descriptive_table(df, output_dir):
    """
    Create a publication-ready descriptive statistics table.
    """
    # Initialize results dictionary
    results = {}

    # Demographics
    n_total = len(df)
    results['Total participants'] = f"{n_total}"

    # Sex distribution
    if 'sex' in df.columns:
        sex_counts = df['sex'].value_counts()
        male_n = sex_counts.get('M', 0) if 'M' in sex_counts else sex_counts.get('male', 0)
        female_n = sex_counts.get('F', 0) if 'F' in sex_counts else sex_counts.get('female', 0)
        male_pct = (male_n / n_total) * 100
        female_pct = (female_n / n_total) * 100
        results['Sex (Male/Female)'] = f"{male_n} ({male_pct:.1f}%) / {female_n} ({female_pct:.1f}%)"

    # Age
    if 'age' in df.columns:
        age_mean = df['age'].mean()
        age_std = df['age'].std()
        age_median = df['age'].median()
        age_q25 = df['age'].quantile(0.25)
        age_q75 = df['age'].quantile(0.75)
        results['Age (years)'] = f"{age_mean:.1f} ± {age_std:.1f} (median: {age_median:.1f}, IQR: {age_q25:.1f}-{age_q75:.1f})"

        # # Save 'participant_id' and 'age' to a separate CSV
        # age_table_path = os.path.join(output_dir, 'participants_age_sex.csv')
        # df[['participant_id', 'age', 'sex']].to_csv(age_table_path, index=False)
        # print(f"Participant ages saved to: {age_table_path}")

    # Time since injury
    if 'mri_time_since_injury' in df.columns:
        tsi_mean = df['mri_time_since_injury'].mean()
        tsi_std = df['mri_time_since_injury'].std()
        tsi_median = df['mri_time_since_injury'].median()
        tsi_q25 = df['mri_time_since_injury'].quantile(0.25)
        tsi_q75 = df['mri_time_since_injury'].quantile(0.75)
        results['Time since injury (days)'] = f"{tsi_mean:.1f} ± {tsi_std:.1f} (median: {tsi_median:.1f}, IQR: {tsi_q25:.1f}-{tsi_q75:.1f})"

    # # AIS grade distribution
    # if 'ais_bl' in df.columns:
    #     ais_counts = df['ais_bl'].value_counts().sort_index()
    #     ais_descriptions = []
    #     for grade, count in ais_counts.items():
    #         pct = (count / n_total) * 100
    #         grade_label = AIS_LABELS.get(grade, grade)
    #         ais_descriptions.append(f"{grade} ({grade_label}): {count} ({pct:.1f}%)")
    #     results['AIS Grade at BL'] = "; ".join(ais_descriptions)

    # Cervical/ThoracoLumbar distribution
    if 'nli_bl' in df.columns:
        nli_counts = df['nli_bl'].value_counts()
        cervical_n = sum(nli_counts.get(level, 0) for level in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
        thoracolumbar_n = sum(nli_counts.get(level, 0) for level in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5'])
        cervical_pct = (cervical_n / n_total) * 100
        thoracolumbar_pct = (thoracolumbar_n / n_total) * 100
        results['Injury region'] = (f"Cervical: {cervical_n} ({cervical_pct:.1f}%); "
                                    f"Thoracolumbar: {thoracolumbar_n} ({thoracolumbar_pct:.1f}%)")

    # # Tetraplegia/Paraplegia distribution
    # if 'tetrapara_bl' in df.columns:
    #     tetra_counts = df['tetrapara_bl'].value_counts()
    #     tetra_n = tetra_counts.get(0, 0)  # 0: tetraplegic
    #     para_n = tetra_counts.get(1, 0)   # 1: paraplegic
    #     tetra_pct = (tetra_n / n_total) * 100
    #     para_pct = (para_n / n_total) * 100
    #     results['Injury level'] = f"Tetraplegia: {tetra_n} ({tetra_pct:.1f}%); Paraplegia: {para_n} ({para_pct:.1f}%)"
    #
    # # Neurological level of injury
    # if 'nli' in df.columns:
    #     nli_counts = df['nli'].value_counts().sort_index()
    #     # Show most common levels
    #     top_nli = nli_counts.head(5)
    #     nli_descriptions = []
    #     for level, count in top_nli.items():
    #         pct = (count / n_total) * 100
    #         nli_descriptions.append(f"{level}: {count} ({pct:.1f}%)")
    #     results['Neurological level (top 5)'] = "; ".join(nli_descriptions)

    # Clinical scores at baseline
    for score_key, score_name in CLINICAL_SCORES_TO_AXES.items():
        bl_col = f'{score_key}_bl'
        if bl_col in df.columns:
            score_data = df[bl_col].dropna()
            if len(score_data) > 0:
                score_mean = score_data.mean()
                score_std = score_data.std()
                score_median = score_data.median()
                score_q25 = score_data.quantile(0.25)
                score_q75 = score_data.quantile(0.75)
                max_possible = CLINICAL_SCORES_MAX[score_key]
                results[f'{score_name} (baseline)'] = f"{score_mean:.1f} ± {score_std:.1f} (median: {score_median:.1f}, IQR: {score_q25:.1f}-{score_q75:.1f}) [max: {max_possible}]"

    # Lesion metrics (using SCT measurements)
    lesion_metrics = {
        'midsagittal_length_sct': 'Midsagittal lesion length (mm)',
        'midsagittal_width_sct': 'Midsagittal lesion width (mm)',
        'total_tissue_bridge_sct': 'Total tissue bridge width (mm)'
    }

    for metric_col, metric_name in lesion_metrics.items():
        if metric_col in df.columns:
            metric_data = df[metric_col].dropna()
            if len(metric_data) > 0:
                metric_mean = metric_data.mean()
                metric_std = metric_data.std()
                metric_median = metric_data.median()
                metric_q25 = metric_data.quantile(0.25)
                metric_q75 = metric_data.quantile(0.75)
                results[metric_name] = f"{metric_mean:.2f} ± {metric_std:.2f} (median: {metric_median:.2f}, IQR: {metric_q25:.2f}-{metric_q75:.2f})"

    # Create DataFrame and save as CSV
    results_df = pd.DataFrame(list(results.items()), columns=['Characteristic', 'Value'])

    # Save to CSV
    table_path = os.path.join(output_dir, 'descriptive_statistics_table.csv')
    results_df.to_csv(table_path, index=False)
    print(f"Descriptive statistics table saved to: {table_path}")

    return results_df


def report_MagneticFieldStrength(df):
    if 'MagneticFieldStrength' in df.columns:
        mfs_data = df['MagneticFieldStrength'].dropna()

        # Merge similar field strengths
        def standardize_field_strength(field):
            if pd.isna(field):
                return 'Unknown'
            elif 1.4 <= field <= 1.6:  # Merge 1.494T with 1.5T
                return '1.5T'
            elif 2.9 <= field <= 3.1:  # Handle 3T variations
                return '3.0T'
            elif 6.9 <= field <= 7.1:  # Handle 7T variations
                return '7.0T'
            else:
                return f'{field:.1f}T'

        standardized_mfs = mfs_data.apply(standardize_field_strength)
        mfs_counts = standardized_mfs.value_counts().sort_index()
        print("\nMagnetic Field Strength distribution:")
        for field, count in mfs_counts.items():
            pct = (count / len(standardized_mfs)) * 100
            print(f"- {field}: {count} ({pct:.1f}%)")


def report_cervical_and_thoracic_injuries(df):
    if 'nli_bl' in df.columns:
        col = df['nli_bl']
        missing_like = (
                col.isna()
                | col.eq(None)
                | col.astype(str).str.fullmatch(r'\s*', na=False)  # empty/whitespace
                | col.astype(str).str.fullmatch(r'(?i)nan|none|nt', na=False)
        )
        # Clean column: convert all missing-like values to np.nan
        nli_data = col.mask(missing_like, np.nan).astype('string')
        # Injury level counts
        cervical_mask = nli_data.str.startswith('C', na=False)
        thoracic_mask = (
                nli_data.str.startswith('T', na=False)
                | nli_data.str.startswith('L', na=False)
        )
        cervical_count = cervical_mask.sum()
        thoracic_count = thoracic_mask.sum()
        print(f'\nNumber of subjects with cervical injuries: {cervical_count}')
        print(f'Number of subjects with thoracic injuries: {thoracic_count}')
        # List participants with missing nli_bl
        missing_nli_ids = df.loc[missing_like, 'participant_id'].tolist()
        if missing_nli_ids:
            print(f'Participants with missing nli_bl after merge: {missing_nli_ids}')
            print(f'Total missing nli_bl: {len(missing_nli_ids)}')


def _make_autopct(values: list[int]):
    """Return a formatter for pie-chart labels with percent and counts."""

    def autopct(pct: float):
        total = sum(values)
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({count})"

    return autopct


def create_comprehensive_figure(df, output_dir):
    """
    Create a comprehensive publication-ready figure with multiple subplots for descriptive analysis.
    """
    # Set style for publication
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = FONT_SIZE
    plt.rcParams['axes.labelsize'] = LABEL_SIZE
    plt.rcParams['xtick.labelsize'] = TICK_SIZE
    plt.rcParams['ytick.labelsize'] = TICK_SIZE

    # Create figure with larger size and tighter spacing
    fig = plt.figure(figsize=(16, 12))

    # Subplot 1: Sex distribution (pie chart)
    ax1 = plt.subplot(2, 2, 1)
    if 'sex' in df.columns:
        sex_counts = df['sex'].value_counts()
        # Handle different sex encodings
        if 'M' in sex_counts.index or 'F' in sex_counts.index:
            labels = ['Male' if x == 'M' else 'Female' for x in sex_counts.index]
        else:
            labels = ['Male' if 'male' in str(x).lower() else 'Female' for x in sex_counts.index]

        wedges, texts, autotexts = ax1.pie(sex_counts.values, labels=labels, autopct=_make_autopct(sex_counts),
                                          colors=PIE_COLORS[:len(sex_counts)], startangle=90,
                                          textprops={'fontsize': TICK_SIZE})
        ax1.set_title('Sex', fontsize=TITLE_SIZE, fontweight='bold')

    # Subplot 2: Age distribution (pie chart by decades)
    ax2 = plt.subplot(2, 2, 2)
    if 'age' in df.columns:
        age_data = df['age'].dropna()

        # Calculate mean and standard deviation
        age_mean = age_data.mean()
        age_std = age_data.std()
        age_min = int(age_data.min())

        # Create age groups by decades
        def age_to_decade(age):
            if pd.isna(age):
                return 'Unknown'
            elif age < 20:
                return f'{age_min}-19'
            elif age < 30:
                return '20-29'
            elif age < 40:
                return '30-39'
            elif age < 50:
                return '40-49'
            elif age < 60:
                return '50-59'
            elif age < 70:
                return '60-69'
            elif age < 80:
                return '70-79'
            else:
                return '80+'

        age_decades = age_data.apply(age_to_decade)
        age_counts = age_decades.value_counts()

        # Sort age groups logically
        decade_order = [f'{age_min}-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
        # Swap the order for clockwise ordering in pie chart
        decade_order = decade_order[::-1]
        sorted_decades = [decade for decade in decade_order if decade in age_counts.index]
        sorted_counts = [age_counts[decade] for decade in sorted_decades]

        # Create pie chart without autopct labels first
        wedges, texts = ax2.pie(sorted_counts, labels=sorted_decades,
                               colors=PIE_COLORS[:len(sorted_decades)], startangle=90,
                               textprops={'fontsize': TICK_SIZE}, labeldistance=1.15)

        # Manually add percentage labels with alternating positions to prevent overlap
        total = sum(sorted_counts)
        for i, (wedge, count) in enumerate(zip(wedges, sorted_counts)):
            # Calculate angle for label placement
            angle = (wedge.theta1 + wedge.theta2) / 2
            # Alternate the distance for every second label to prevent overlap
            if i == 0:
                distance = 0.85  # Further from center
            else:
                distance = 0.6  # Closer to center

            # Convert angle to radians
            angle_rad = np.radians(angle)
            # Calculate position
            x = distance * np.cos(angle_rad)
            y = distance * np.sin(angle_rad)
            # Calculate percentage
            percentage = (count / total) * 100

            # Add text with background for better visibility
            ax2.text(x, y, f'{percentage:.1f}%\n({count})',
                    ha='center', va='center', fontsize=TICK_SIZE)
                    # bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                    #          edgecolor="gray", alpha=0.8))

        ax2.set_title('Age', fontsize=TITLE_SIZE, fontweight='bold')

        # Add mean (SD) age below the pie chart
        ax2.text(0.5, -0.05, f'Mean (SD): {age_mean:.1f} ({age_std:.1f}) years',
                ha='center', va='center', fontsize=LABEL_SIZE,
                transform=ax2.transAxes)

    # Subplot 3: AIS grade distribution across time (stacked bar chart)
    ax3 = plt.subplot(2, 2, 3)

    # Define time points for AIS grades
    ais_time_points = ['bl', '1m', '3m', '6m']

    # Check which AIS time points have data
    available_ais_timepoints = []
    for tp in ais_time_points:
        ais_col = f'ais_{tp}'
        if ais_col in df.columns and not df[ais_col].dropna().empty:
            available_ais_timepoints.append(tp)

    if available_ais_timepoints:
        # Get all unique AIS grades across all time points, excluding NT values
        all_grades = set()
        for tp in available_ais_timepoints:
            ais_col = f'ais_{tp}'
            grades = df[ais_col].dropna()
            # Filter out NT values
            grades = grades[grades != 'NT']
            all_grades.update(grades.unique())

        # Sort grades (A, B, C, D, E, then any others)
        grade_order = ['A', 'B', 'C', 'D', 'E']
        sorted_grades = [g for g in grade_order if g in all_grades]
        sorted_grades.extend([g for g in sorted(all_grades) if g not in grade_order])

        # Prepare data for stacked bar chart
        tp_labels = [tp.upper() if tp != 'bl' else 'BL' for tp in available_ais_timepoints]
        grade_counts = {grade: [] for grade in sorted_grades}
        total_counts = []

        for tp in available_ais_timepoints:
            ais_col = f'ais_{tp}'
            tp_data = df[ais_col].dropna()
            tp_total = len(tp_data)
            total_counts.append(tp_total)

            tp_counts = tp_data.value_counts()
            for grade in sorted_grades:
                count = tp_counts.get(grade, 0)
                grade_counts[grade].append(count)

        # Create stacked bar chart
        bottom = np.zeros(len(available_ais_timepoints))
        x_pos = range(len(available_ais_timepoints))

        bars = []
        for i, grade in enumerate(sorted_grades):
            # Use the same colormap as other subplots
            color = AIS_COLORS[i % len(AIS_COLORS)]
            label = f"AIS {grade}" if grade in AIS_LABELS else f"AIS {grade}"
            bar = ax3.bar(x_pos, grade_counts[grade], bottom=bottom,
                         color=color, alpha=1, label=label,
                         edgecolor='white', linewidth=0.5)
            bars.append(bar)

            # Add count labels for each sub-bar (only if count > 0)
            for j, (x, count) in enumerate(zip(x_pos, grade_counts[grade])):
                if count > 0:  # Only show label if there are participants
                    y_center = bottom[j] + count / 2  # Center of the sub-bar
                    ax3.text(x, y_center, str(count), ha='center', va='center',
                            fontsize=TICK_SIZE, fontweight='bold', color='black')

            bottom += grade_counts[grade]

        # Customize the plot
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(tp_labels, fontsize=TICK_SIZE)
        ax3.set_xlabel('Time Point', fontsize=LABEL_SIZE)
        ax3.set_ylabel('Number of Participants', fontsize=LABEL_SIZE)
        ax3.set_title('AIS Grade Distribution Over Time', fontsize=TITLE_SIZE, fontweight='bold')
        ax3.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

        # Add legend at right center (0.78, 0.42)
        ax3.legend(bbox_to_anchor=(1, 0.9), loc='center left', fontsize=TICK_SIZE-2, framealpha=0.9)

        # Remove the total sample size annotations above bars since we now show counts within sub-bars

        # Remove right and top spines
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)

        # Set y-axis to start from 0
        ax3.set_ylim(0, max(total_counts) * 1.1)

    else:
        # Fallback: show only baseline AIS grade distribution as pie chart if no longitudinal data
        if 'ais_bl' in df.columns:
            # Filter out NT values
            ais_data = df['ais_bl'][df['ais_bl'] != 'NT']
            ais_counts = ais_data.value_counts().sort_index()
            labels = []
            for grade in ais_counts.index:
                if grade in AIS_LABELS:
                    labels.append(f"AIS {grade}\n({AIS_LABELS[grade]})")
                else:
                    labels.append(f"AIS {grade}")
            wedges, texts, autotexts = ax3.pie(ais_counts.values, labels=labels, autopct='%1.1f%%',
                                              colors=PIE_COLORS[:len(ais_counts)], startangle=90,
                                              textprops={'fontsize': TICK_SIZE})
            ax3.set_title('AIS Grade (Baseline Only)', fontsize=TITLE_SIZE, fontweight='bold')

    # Subplot 4: Neurological level of injury
    ax4 = plt.subplot(2, 2, 4)
    if 'nli_bl' in df.columns:
        col = df['nli_bl']

        # Missing-like values
        missing_like = (
                col.isna()
                | col.eq(None)
                | col.astype(str).str.fullmatch(r'\s*', na=False)
                | col.astype(str).str.fullmatch(r'(?i)nan|none|nt', na=False)
        )
        # Convert missing-like values to the label "Unknown"
        nli_data = col.astype('string').mask(missing_like, 'Unknown')
        # Count values
        nli_counts = nli_data.value_counts()

        def sort_nli(level: str) -> tuple[int, int]:
            """Sort neurological levels anatomically: C1-C8, T1-T12, L1-L5, S1-S5, Unknown.
            Args:
                level: NLI string like 'C5'.
            Returns:
                A tuple for anatomical sorting.
            """
            if level == 'Unknown':
                return (999, 999)
            if len(level) >= 2 and level[0].isalpha() and level[1:].isdigit():
                letter = level[0].upper()
                number = int(level[1:])
                letter_order = {'C': 1, 'T': 2, 'L': 3, 'S': 4}
                return (letter_order.get(letter, 998), number)
            return (999, 999)

        sorted_levels = sorted(nli_counts.index, key=sort_nli)
        sorted_counts = [nli_counts[level] for level in sorted_levels]

        bars = ax4.bar(
            range(len(sorted_levels)),
            sorted_counts,
            color=PIE_COLORS[1],
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5,
        )
        ax4.set_xticks(range(len(sorted_levels)))
        ax4.set_xticklabels(sorted_levels, rotation=45, fontsize=TICK_SIZE)
        ax4.set_xlabel('Neurological Level', fontsize=LABEL_SIZE)
        ax4.set_ylabel('Number of Participants', fontsize=LABEL_SIZE)
        ax4.set_title('Neurological Level of Injury at BL', fontsize=TITLE_SIZE, fontweight='bold')
        ax4.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)

        if len(sorted_levels) <= 20:
            for bar, value in zip(bars, sorted_counts):
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(value),
                    ha='center',
                    va='bottom',
                    fontsize=TICK_SIZE,
                )

        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

    # Use tighter layout with minimal padding
    plt.tight_layout(pad=1.5, h_pad=1.0, w_pad=1.0)

    # Save the figure
    figure_path = os.path.join(output_dir, 'Fig2_descriptive_analysis_comprehensive.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(figure_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')

    print(f"Comprehensive descriptive figure saved to: {figure_path}")
    plt.close()


def main():
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Create output directory
    output_dir = args.o
    os.makedirs(output_dir, exist_ok=True)

    print("\nReading data files...")
    # -----------
    # CSV file with lesion metrics and clinical scores for sci-zurich
    # Note: we don't use the lesion metrics but need participant_id column
    # -----------
    df = read_csv_file_with_lesion_metrics(args.i)
    # Keep only relevant columns: participant_id, midsagittal_length_sct, midsagittal_width_sct, total_tissue_bridge_sct
    # NOTE: although this file also contains clinical scores, we will read clinical data from separate files
    df = df[['participant_id',
             'session_id',
             'midsagittal_length_sct',
             'midsagittal_width_sct',
             'total_tissue_bridge_sct']]

    # -----------
    # participants.tsv files for both datasets with sex, age, MagneticFieldStrength
    # -----------
    df_participants_zurich = read_participants_file(args.file_participants_zurich)
    df_participants_nisci = read_participants_file(args.file_participants_nisci)

    # --------------
    # sci-zurich clinical data
    # --------------
    df_clinical_sci_zurich = pd.read_excel(args.file_clinical_sci_zurich, engine='openpyxl',
                                         usecols=['participant_id', 'session_id', 'nli_bl',
                                                  'ais_bl', 'ais_1m', 'ais_3m', 'ais_6m'])
    # If session_id is empty, fill with 'ses-01'
    df_clinical_sci_zurich['session_id'] = df_clinical_sci_zurich['session_id'].fillna('ses-01')
    # Replace 'NT' with NaN
    df_clinical_sci_zurich = df_clinical_sci_zurich.replace('NT', np.nan)
    df = pd.merge(df, df_clinical_sci_zurich, on=['participant_id', 'session_id'], how='left')

    # -----------
    # XLSX file with clinical scores (NLI, AIS) for NISCI
    # -----------
    df_clinical_nisci = pd.read_excel(args.file_clinical_nisci, engine='openpyxl',
                                      usecols=['Patient', 'NLI_01', 'AIS_01', 'AIS_03', 'AIS_05', 'AIS_06'])
    # '01' -- Day 0 (Screening)
    # '02' -- Day 1 (Baseline)
    # '03' -- 2 Weeks (14 days)
    # '04' -- 1 month (30 days)
    # '05' -- 3 months (84 days)
    # '06' -- 6 months (168 days)
    df_clinical_nisci = df_clinical_nisci.rename(columns={'Patient': 'participant_id',
                                                          'NLI_01': 'nli_bl',
                                                          'AIS_01': 'ais_bl',       # using '01' as baseline as there's some missing data in '02'
                                                          'AIS_03': 'ais_1m',       # for details see nisci-trial/README.md
                                                          'AIS_05': 'ais_3m',
                                                          'AIS_06': 'ais_6m'})

    # -----------
    # Combine participant demographics from both datasets (add rows to a single dataframe)
    # -----------
    df_participants_zurich['source'] = 'sci-zurich'
    df_participants_zurich['Site'] = 'Zurich'
    df_participants_nisci['source'] = 'nisci'
    df_participants_merged = pd.concat([df_participants_zurich, df_participants_nisci], ignore_index=True)

    # -----------
    # Merge the dataframes
    # -----------
    print("\nMerging dataframes...")
    df = pd.merge(df, df_participants_merged, on='participant_id', how='left')
    # Merge nli_bl and ais columns
    # NOTE: these columns already exist in df from sci-zurich, so we only add missing values from nisci
    df = pd.merge(df, df_clinical_nisci, on='participant_id', how='left', suffixes=('', '_nisci'))
    # Fill missing nli_bl values from nisci
    df['nli_bl'] = df['nli_bl'].combine_first(df['nli_bl_nisci'])
    df['ais_bl'] = df['ais_bl'].combine_first(df['ais_bl_nisci'])
    df['ais_1m'] = df['ais_1m'].combine_first(df['ais_1m_nisci'])
    df['ais_3m'] = df['ais_3m'].combine_first(df['ais_3m_nisci'])
    df['ais_6m'] = df['ais_6m'].combine_first(df['ais_6m_nisci'])
    # Drop the extra columns
    df = df.drop(columns=['nli_bl_nisci', 'ais_bl_nisci', 'ais_1m_nisci', 'ais_3m_nisci', 'ais_6m_nisci'])

    report_MagneticFieldStrength(df)

    # Print number of subjects cervical and thoracic injuries
    report_cervical_and_thoracic_injuries(df)

    # # Drop rows with NaN values in key lesion metrics
    # lesion_metrics = ['midsagittal_length_sct', 'midsagittal_width_sct', 'total_tissue_bridge_sct']
    # available_metrics = [metric for metric in lesion_metrics if metric in df.columns]
    # if available_metrics:
    #     df = df.dropna(subset=available_metrics)
    #     print(f'Number of subjects after dropping NaN values in lesion metrics: {df.shape[0]}')

    # Create descriptive statistics table
    print("\nCreating descriptive statistics table...")
    create_descriptive_table(df, output_dir)

    # Create comprehensive figure
    print("\nCreating comprehensive descriptive figure...")
    create_comprehensive_figure(df, output_dir)

    # Print summary statistics
    print(f"\nSummary:")
    print(f"- Total subjects analyzed: {len(df)}")
    print(f"- Descriptive table saved as CSV")
    print(f"- Comprehensive figure created")
    print(f"- All outputs saved to: {output_dir}")

    print("\nDescriptive analysis completed successfully!")

if __name__ == '__main__':
    main()
