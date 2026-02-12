#!/usr/bin/env python
#
# Create AIS grade distribution and neurological level of injury distribution figures for URP-CTREE analysis (Figure 4
# and Supplementary Figure 1)
#
# For each URP-CTREE node, creates two barplots:
# 1. AIS grade distribution (ais_bl)
# 2. Neurological level of injury distribution with Cervical/ThoracoLumbar categories (nli_bl)
#


import os
import argparse
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Constants for plotting
TITLE_SIZE = 16
LABEL_SIZE = FONT_SIZE = TITLE_SIZE - 2
TICK_SIZE = TITLE_SIZE

# Color scheme based on the existing analysis (order: Unknown, A, B, C, D, E)
AIS_COLORS = [
    '#E5D8BD',  # Unknown – light beige/gray
    '#FBB4AE',  # A – pastel red/pink
    '#FED9A6',  # B – pastel orange
    '#FFFFCC',  # C – pastel yellow
    '#CCEBC5',  # D – pastel green
    '#B3CDE3',  # E – cool green-blue
]

PIE_COLORS = ['#FBB4AE', '#B3CDE3', '#CCEBC5', '#DECBE4', '#FED9A6', '#FFFFCC', '#E5D8BD', '#FDDAEC', '#F2F2F2']


def get_parser():
    """
    Parser function for command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Create AIS grade distribution and neurological level of injury distribution figures for URP-CTREE analysis.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        metavar="<file>",
        required=True,
        type=str,
        help='Path to input CSV file containing ais_bl, nodeProCSM, and nli_bl columns'
    )
    parser.add_argument(
        '-o',
        metavar="<folder>",
        required=True,
        type=str,
        help='Path to output folder for saving figures'
    )

    return parser


def create_ais_distribution_plot(df_subset, node_value, output_dir):
    """
    Create AIS grade distribution stacked barplot for a specific nodeProCSM value.
    """
    # Set style for publication
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = FONT_SIZE

    fig, ax = plt.subplots(figsize=(4, 8))

    # Get AIS baseline data
    if 'ais_bl' in df_subset.columns:
        ais_data = df_subset['ais_bl'].replace('NaN', pd.NA).fillna('Unknown')
        ais_counts = ais_data.value_counts()

        # Define order for stacking: E at bottom, Unknown at top (matching 05_descriptive_analysis.py)
        grade_order_for_stacking = ['E', 'D', 'C', 'B', 'A', 'Unknown']
        # Define order for legend: Unknown, A, B, C, D, E
        grade_order_for_legend = ['Unknown', 'A', 'B', 'C', 'D', 'E']

        # Color mapping matching 05_descriptive_analysis.py legend
        grade_colors = {
            'Unknown': '#E5D8BD',  # Unknown – light beige/gray
            'A': '#FBB4AE',        # A – pastel red/pink
            'B': '#FED9A6',        # B – pastel orange
            'C': '#FFFFCC',        # C – pastel yellow
            'D': '#CCEBC5',        # D – pastel green
            'E': '#B3CDE3'         # E – cool green-blue
        }

        # Use stacking order for building the chart
        sorted_grades = [g for g in grade_order_for_stacking if g in ais_counts.index]
        sorted_counts = [ais_counts[grade] for grade in sorted_grades]

        # Create colors for the grades present
        colors = [grade_colors[grade] for grade in sorted_grades]

        # Create stacked bar chart (single bar)
        bottom = 0
        bars = []
        x_pos = 0  # Single bar at position 0

        for i, (grade, count, color) in enumerate(zip(sorted_grades, sorted_counts, colors)):
            bar = ax.bar(x_pos, count, bottom=bottom, color=color, alpha=0.8,
                        edgecolor='white', linewidth=0.5,
                        label=f"AIS {grade}" if grade != 'Unknown' else "Unknown")
            bars.append(bar)

            # Add count labels for each segment (only if count > 0)
            if count > 0:
                if grade == 'Unknown' and count == 3:  # Special case
                    y_center = bottom + count / 2 - 0.8  # Shift down slightly
                else:
                    y_center = bottom + count / 2
                ax.text(x_pos, y_center, f'{grade}\n(n = {count})',
                        ha='center', va='center',
                       fontsize=TICK_SIZE, fontweight='bold', color='black')

            bottom += count

        # Customize the plot
        ax.set_xticks([x_pos])
        # ax.set_xticklabels([node_value], fontsize=TICK_SIZE)
        # ax.set_ylabel('Number of Participants', fontsize=LABEL_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        # Show no x- and y-ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # # Add legend with proper display order (Unknown, A, B, C, D, E)
        # legend_handles = []
        # legend_labels = []
        # for grade in grade_order_for_legend:
        #     if grade in ais_counts.index:  # Only include grades that exist in the data
        #         handle = mpatches.Patch(color=grade_colors[grade],
        #                               label=f"AIS {grade}" if grade != 'Unknown' else "Unknown")
        #         legend_handles.append(handle)
        #         legend_labels.append(f"AIS {grade}" if grade != 'Unknown' else "Unknown")
        #
        # ax.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.05, 1),
        #          loc='upper left', fontsize=TICK_SIZE-2, framealpha=0.9)

        # Remove top and right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Set y-axis to start from 0
        ax.set_ylim(0, bottom * 1.05 if bottom > 0 else 1)

    plt.tight_layout()

    # Save figure
    output_filename = f"ais_distribution_{node_value}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved AIS distribution plot: {output_path}")


def create_nli_distribution_plot(df_subset, node_value, output_dir):
    """
    Create neurological level of injury distribution stacked barplot for a specific nodeProCSM value.
    Shows Cervical vs ThoracoLumbar distribution.
    """
    # Set style for publication
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = FONT_SIZE

    fig, ax = plt.subplots(figsize=(4, 8))

    # Get NLI baseline data
    if 'nli_bl' in df_subset.columns:
        nli_data = df_subset['nli_bl'].dropna()
        nli_counts = nli_data.value_counts()

        # Calculate cervical and thoracolumbar counts
        cervical_levels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
        thoracolumbar_levels = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
                               'L1', 'L2', 'L3', 'L4', 'L5']

        cervical_n = sum(nli_counts.get(level, 0) for level in cervical_levels)
        thoracolumbar_n = sum(nli_counts.get(level, 0) for level in thoracolumbar_levels)

        total = cervical_n + thoracolumbar_n

        if total > 0:
            # Order for stacking: ThoracoLumbar at bottom, Cervical at top
            categories_stacking = ['ThoracoLumbar', 'Cervical']
            counts_stacking = [thoracolumbar_n, cervical_n]
            # Order for legend: Cervical, ThoracoLumbar (same as before)
            categories_legend = ['Cervical', 'ThoracoLumbar']
            colors = [PIE_COLORS[0], PIE_COLORS[1]]  # Cervical=red, ThoracoLumbar=blue

            # Create color mapping
            color_map = {'Cervical': PIE_COLORS[0], 'ThoracoLumbar': PIE_COLORS[1]}

            # Create stacked bar chart (single bar)
            x_pos = 0  # Single bar at position 0
            bottom = 0

            for category, count in zip(categories_stacking, counts_stacking):
                if count > 0:
                    bar = ax.bar(x_pos, count, bottom=bottom, color=color_map[category], alpha=0.8,
                               edgecolor='white', linewidth=0.5, label=category)
                    if category == 'ThoracoLumbar' and count == 2:  # Special case
                        y_center = bottom + count / 2 + 0.8  # Shift up
                    else:
                        y_center = bottom + count / 2
                    ax.text(x_pos, y_center, f'{category}\n(n = {count})',
                           ha='center', va='center', fontsize=TICK_SIZE, fontweight='bold')

                    bottom += count

        # Customize the plot
        ax.set_xticks([x_pos])
        # ax.set_xticklabels([node_value], fontsize=TICK_SIZE)
        # ax.set_ylabel('Number of Participants', fontsize=LABEL_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        # Show no x- and y-ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # # Add legend with proper display order (Cervical, ThoracoLumbar)
        # if total > 0:
        #     import matplotlib.patches as mpatches
        #     legend_handles = []
        #     legend_labels = []
        #     for category in categories_legend:
        #         if (category == 'Cervical' and cervical_n > 0) or (category == 'ThoracoLumbar' and thoracolumbar_n > 0):
        #             handle = mpatches.Patch(color=color_map[category], label=category)
        #             legend_handles.append(handle)
        #             legend_labels.append(category)
        #
        #     ax.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.05, 1),
        #              loc='upper left', fontsize=TICK_SIZE-2, framealpha=0.9)

        # Remove top and right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Set y-axis to start from 0
        ax.set_ylim(0, total * 1.05 if total > 0 else 1)

    plt.tight_layout()

    # Save figure
    output_filename = f"nli_distribution_{node_value}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved NLI distribution plot: {output_path}")


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.i):
        raise ValueError(f'ERROR: {args.i} does not exist.')

    # Create output directory if it doesn't exist
    output_dir = os.path.abspath(args.o)
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV file
    try:
        df = pd.read_csv(args.i)
        print(f"Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns.")
    except Exception as e:
        raise ValueError(f"ERROR: Could not read CSV file {args.i}. {str(e)}")

    # Validate required columns
    required_columns = ['ais_bl', 'nodeProCSM', 'nli_bl']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"ERROR: Missing required columns: {missing_columns}")

    # Get unique nodeProCSM values
    node_values = df['nodeProCSM'].dropna().unique()
    print(f"Found {len(node_values)} unique nodeProCSM values: {list(node_values)}")

    # Process each nodeProCSM value
    for node_value in node_values:
        print(f"\nProcessing {node_value}...")

        # Filter data for current nodeProCSM value
        df_subset = df[df['nodeProCSM'] == node_value].copy()
        print(f"  - {len(df_subset)} participants for {node_value}")

        # Create AIS distribution plot
        create_ais_distribution_plot(df_subset, node_value, output_dir)

        # Create NLI distribution plot
        create_nli_distribution_plot(df_subset, node_value, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
