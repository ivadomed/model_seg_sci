import pandas as pd
import argparse


sites_to_rename = {
    'site-01 \n(non-traumatic SCI)': 'site-01\n(DCM)',
    'site-01 \n(traumatic SCI)': 'site-01\n(tSCI)',
    'site-02 \n(traumatic SCI)': 'site-02\n(tSCI)', 
    'site-04 \n(acute traumatic SCI)': 'site-04\n(acuteSCI)', 
    'site-07 \n(acute traumatic SCI)': 'site-07\n(acuteSCI)'
}

metrics_to_rename = {
    'nsd_mean': 'NSD',
    'rel_vol_error_mean': 'RVE',
    'lesion_ppv_mean': 'PPVL',
    'lesion_sensitivity_mean': 'SensL',
    'lesion_f1_score_mean': 'F1ScoreL'
}

models_to_rename = {
    'SCIsegV2Multi\n_AggressiveDA': 'SCIsegV2\nmulti\naggressiveDA',
    'SCIsegV2Multi\n_OriginalDA': 'SCIsegV2\nmulti\ndefaultDA',
    'SCIsegV2Region\n_AggressiveDA': 'SCIsegV2\nsingle\naggressiveDA',
    'SCIsegV2Region\n_OriginalDA': 'SCIsegV2\nsingle\ndefaultDA',
}

def get_parser():
    parser = argparse.ArgumentParser(description='Generate table 1 for the paper')
    parser.add_argument('-i', type=str, required=True, help='Path to the CSV file with the metrics')
    return parser


# Helper function to create rows for each model
def create_rows(model, df, metrics_mean_list, metrics_std_list):
    rows = ""
    len_metrics = len(metrics_mean_list)
    wrapped_model = r"\makecell{" + model.replace('\n', r' \\ ') + r"}"
    for i in range(len_metrics):
        if i == 0:
            rows += f"\\multirow{{{len_metrics}}}{{*}}{{{wrapped_model}}} & {metrics_to_rename[metrics_mean_list[i]]} "
        else:
            rows += f" & {metrics_to_rename[metrics_mean_list[i]]} "
        
        for site in sites_to_rename.values():
            mean = df.loc[(df['model'] == model) & (df['site'] == site), metrics_mean_list[i]].values[0]
            std = df.loc[(df['model'] == model) & (df['site'] == site), metrics_std_list[i]].values[0]
            rows += f"& {mean:.2f} $\pm$ {std:.2f} "
        rows += "\\\\\n"
    return rows


def main():
    args = get_parser().parse_args()
    df = pd.read_csv(args.i)

    # # Create LaTeX table
    # latex_table = r"""
    # \begin{table}[htbp]
    #     \centering
    #     \begin{tabular}{|l|l|c|c|c|c|c|}
    #         \hline
    #         \multirow{3}{*}{\textbf{Model}} & \multirow{3}{*}{\textbf{Metric}} & \multicolumn{5}{c|}{\textbf{Sites}} \\
    #         \cline{3-7} &
    #         & \multirow{2}{*}{\textbf{\makecell{site-01 \\ (DCM)}}} & \textbf{site-01 \\ (tSCI)} & \textbf{site-02 \\ (tSCI)} & \textbf{site-04 \\ (acuteSCI)} & \textbf{site-07 \\ (acuteSCI)} \\
    #         \hline
    # """

    # Create LaTeX table
    latex_table = r"""
    \begin{table}[htbp]
        \centering
        \setlength{\tabcolsep}{5pt} % Adjust the length as needed
        \caption{Metrics per site for different models}
        \resizebox{\textwidth}{!}{%
        \begin{tabular}{llccccc}
        \toprule
            \multirow{3}{*}{\textbf{Model}} & \multirow{3}{*}{\textbf{Metric}} & \multicolumn{5}{c}{\textbf{Test Sites}} \\
            \cline{3-7} & & 
            \multirow{2}{*}{\textbf{\makecell{site-01 \\ (DCM)}}} & \multirow{2}{*}{\textbf{\makecell{site-01 \\ (tSCI)}}} & \multirow{2}{*}{\textbf{\makecell{site-02 \\ (tSCI)}}} & \multirow{2}{*}{\textbf{\makecell{site-04 \\ (acuteSCI)}}} & \multirow{2}{*}{\textbf{\makecell{site-07 \\ (acuteSCI)}}} \\
            \\
            \hline
    """

    # drop column with 'rel_vol_error'
    df.drop(columns=['rel_vol_error_mean', 'rel_vol_error_std'], inplace=True)

    metrics_mean_list = df.columns[df.columns.str.contains('mean')].tolist()
    metrics_std_list = df.columns[df.columns.str.contains('std')].tolist()

    # Rename sites
    df['site'] = df['site'].map(sites_to_rename)
    # Rename models
    df['model'] = df['model'].map(models_to_rename)


    # Generate rows for each model
    for model in df['model'].unique():
        latex_table += create_rows(model, df, metrics_mean_list, metrics_std_list)
        latex_table += "\\hline\n"

    latex_table += r"""
        \bottomrule
        \end{tabular}%
        }
        \label{tab:metrics}
    \end{table}
    """

    print(latex_table)

if __name__ == '__main__':
    main()