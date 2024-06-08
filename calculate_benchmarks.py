import os
import pandas as pd
from Trojan_Mapping import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import colorcet as cc

BENCHS_OUTPUT_FOLDER = "output"
CALC_OUTPUT_FOLDER = "bench_calc"  # for our calculations in this file
BENCHS_TO_CALC = ['RS232-T1100-uart',
                  'RS232-T1200-uart',
                  'RS232-T1300-uart',
                  'RS232-T1400-uart',
                  'RS232-T1500-uart',
                  'RS232-T1600-uart',
                  's35932-T100_scan_free',
                  's35932-T100_scan_in',
                  's35932-T200_scan_free',
                  's35932-T200_scan_in',
                  's35932-T300_scan_free',
                  's35932-T300_scan_in',
                  #'s38584-T100_scan_free',
                  #'s38584-T100_scan_in',
                  #'s38584-T200_scan_free',
                  #'s38584-T200_scan_in',
                  'b01',
                  'b02',
                  'b03',
                  'b04',
                  'b05',
                  'b06',
                  'b07',
                  'b08',
                  'b09',
                  'b10',
                  'b12',
                  'b13',
                  'b30',
                  #'c499',
                  'c880',
                  #'c1355',
                  #'c1908',
                  'c5315'
]


def find_required_dfs_lengths_files(bench_folder):
    """
    scans the files in a ../output/bench_name path and finds the ones with dfs depths of 2, 4, inf
    :param bench_folder:  ../output/bench_name path which contains the results of our HT_detection code
    :return: a tuple with the paths of the corresponding csv files
    """
    lim2_csv, lim4_csv, inf_csv = '', '', ''
    for entry in os.listdir(bench_folder):
        entry_path = os.path.join(bench_folder, entry)
        if os.path.isfile(entry_path) and entry.endswith('.csv'):
            if 'inf' in entry:
                inf_csv = entry_path
            elif 'lim2' in entry:
                lim2_csv = entry_path
            elif 'lim4' in entry:
                lim4_csv = entry_path
    return lim2_csv, lim4_csv, inf_csv


def handle_bench_calculation(bench_folder):
    """
    calculating the metrics for our paper, using the corresponding data frames
    :param bench_folder:  ../output/bench_name path which contains the results of our HT_detection code
    """

    # checking if required save paths exists
    if not os.path.exists(CALC_OUTPUT_FOLDER):
        os.makedirs(CALC_OUTPUT_FOLDER)
    sub_dir = os.path.join(CALC_OUTPUT_FOLDER, os.path.basename(bench_folder))
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # getting the data frames we want to work on
    lim2_csv, lim4_csv, inf_csv = find_required_dfs_lengths_files(bench_folder)
    lim2_df = pd.read_csv(lim2_csv).reset_index()
    lim4_df = pd.read_csv(lim4_csv).reset_index()
    inf_df = pd.read_csv(inf_csv).reset_index()

    # defining our desired df for our old fanci metric calculation
    calc_df_old_fanci = pd.DataFrame(columns=['GATE_NAME', 'GATE_TYPE',
                                              'LIM2_MEAN', 'LIM4_MEAN', 'INF_MEAN',
                                              'LIM2_MEDIAN', 'LIM4_MEDIAN', 'INF_MEDIAN',
                                              'LIM2_SIZE', 'LIM4_SIZE', 'INF_SIZE'])

    calc_df_old_fanci[['GATE_NAME', 'GATE_TYPE']] = lim2_df[['GATE_NAME', 'GATE_TYPE']]
    calc_df_old_fanci[['LIM2_MEAN', 'LIM2_MEDIAN', 'LIM2_SIZE']] = lim2_df[['MEAN', 'MEDIAN', 'SIZE']]
    calc_df_old_fanci[['LIM4_MEAN', 'LIM4_MEDIAN', 'LIM4_SIZE']] = lim4_df[['MEAN', 'MEDIAN', 'SIZE']]
    calc_df_old_fanci[['INF_MEAN', 'INF_MEDIAN', 'INF_SIZE']] = inf_df[['MEAN', 'MEDIAN', 'SIZE']]

    # saving the df inside bench_calc\bench_name folder
    calc_df_old_fanci.to_csv(os.path.join(sub_dir, f'{os.path.basename(sub_dir)}_fanci_per_lim.csv'))

    # defining another df, now with a different calculation method
    calc_df_fanci_dist = pd.DataFrame(columns=['GATE_NAME', 'GATE_TYPE',
                                               'INF-LIM2_MEAN', 'INF-LIM4_MEAN', 'INF-LIM2_NORM_MEAN', 'INF-LIM4_NORM_MEAN',
                                               'INF-LIM2_MEDIAN', 'INF-LIM4_MEDIAN', 'INF-LIM2_NORM_MEDIAN', 'INF-LIM4_NORM_MEDIAN',
                                               'INF-LIM2_SIZE', 'INF-LIM4_SIZE'])

    calc_df_fanci_dist[['GATE_NAME', 'GATE_TYPE']] = lim2_df[['GATE_NAME', 'GATE_TYPE']]
    calc_df_fanci_dist[['INF-LIM2_MEAN', 'INF-LIM2_MEDIAN']] = abs(lim2_df[['MEAN', 'MEDIAN']] - inf_df[['MEAN', 'MEDIAN']]) / inf_df[['MEAN', 'MEDIAN']]
    calc_df_fanci_dist[['INF-LIM2_NORM_MEAN', 'INF-LIM2_NORM_MEDIAN']] = abs(lim2_df[['MEAN', 'MEDIAN']] - inf_df[['MEAN', 'MEDIAN']]) / lim2_df[['MEAN', 'MEDIAN']]
    calc_df_fanci_dist[['INF-LIM2_SIZE']] = inf_df[['SIZE']] - lim2_df[['SIZE']]

    calc_df_fanci_dist[['INF-LIM4_MEAN', 'INF-LIM4_MEDIAN']] = abs(lim4_df[['MEAN', 'MEDIAN']] - inf_df[['MEAN', 'MEDIAN']]) / inf_df[['MEAN', 'MEDIAN']]
    calc_df_fanci_dist[['INF-LIM4_NORM_MEAN', 'INF-LIM4_NORM_MEDIAN']] = abs(lim4_df[['MEAN', 'MEDIAN']] - inf_df[['MEAN', 'MEDIAN']]) / lim4_df[['MEAN', 'MEDIAN']]
    calc_df_fanci_dist[['INF-LIM4_SIZE']] = inf_df[['SIZE']] - lim4_df[['SIZE']]
    calc_df_fanci_dist.replace([np.nan], np.inf, inplace=True)  # "(0-0)/0" cases defined as inf

    # saving the df inside bench_calc\bench_name folder
    calc_df_fanci_dist.to_csv(os.path.join(sub_dir, f'{os.path.basename(sub_dir)}_fanci_dist_over_lims.csv'))

    return calc_df_old_fanci, calc_df_fanci_dist


def calc_proportions_old_method(df, threshold, netlist_file, header):
    """
    calculates the fp and fn rates according to the old paper implementation
    :param df: the dataframe we calculate the threshold on
    :param threshold: threshold for mean/median fanci value
    :param netlist_file: for trojan mapping
    :param header: the header in the df we want to apply the threshold to
    :return: the fp and fn rates. fn is a "binary" parameter
    """
    fp_rate = 0
    detected_trojan = ""

    # calc rows that fulfil the condition
    filtered_df = df[df[header] <= threshold]

    # List of pre-defined names to filter out
    gates_to_exclude = []
    if netlist_file in trojan_mapping.keys():
        gates_to_exclude = list(trojan_mapping[netlist_file])
    else:
        detected_trojan = "no trojan"

    # Filter out rows based on the pre-defined list of names
    filtered_df_excluded = filtered_df[~filtered_df['GATE_NAME'].isin(gates_to_exclude)]

    # check if even found trojan gates at filtered_df (questionable implementation)
    if detected_trojan != "no trojan":
        if len(filtered_df_excluded) == len(filtered_df):
            detected_trojan = "no"
        else:
            detected_trojan = "yes"

    # calculates fp rate
    fp_rate = (len(filtered_df_excluded) / float(len(df))) * 100

    return fp_rate, detected_trojan


def calc_proportions_new_method(df, threshold, netlist_file, header):
    """
    calculates the fp and fn rates according to our new dfs distance evaluation. the condition is now >= threshold
    :param df: the dataframe we calculate the threshold on
    :param threshold: threshold for mean/median dfs dist value
    :param netlist_file: for trojan mapping
    :param header: the header in the df we want to apply the threshold to
    :return: the fp and fn rates. fn is a "binary" parameter
    """
    fp_rate = 0
    detected_trojan = ""

    # calc rows that fulfil the condition
    filtered_df = df[df[header] >= threshold]

    # List of pre-defined names to filter out
    gates_to_exclude = []
    if netlist_file in trojan_mapping.keys():
        gates_to_exclude = list(trojan_mapping[netlist_file])
    else:
        detected_trojan = "no trojan"

    # Filter out rows based on the pre-defined list of names
    filtered_df_excluded = filtered_df[~filtered_df['GATE_NAME'].isin(gates_to_exclude)]

    # check if even found trojan gates at filtered_df (questionable implementation)
    if detected_trojan != "no trojan":
        if len(filtered_df_excluded) == len(filtered_df):
            detected_trojan = "no"
        else:
            detected_trojan = "yes"

    fp_rate = (len(filtered_df_excluded) / float(len(df))) * 100

    return fp_rate, detected_trojan


# for curve fitting
def model_function_exp(x, a, b, c):
    return a * np.exp(-b * np.array(x)) - c


def model_function_lin(x, a, b):
    return a * np.array(x) + b


def df_to_latex(df, output_tex_file):
    # Read CSV file into a pandas DataFrame

    # Generate LaTeX code for the table
    latex_code = df.to_latex(index=False)

    # Write LaTeX code to a file
    with open(output_tex_file, 'w') as f:
        f.write(latex_code)


# not the most elegant but it would do
class Figure:
    def __init__(self,
                 suptitle,
                 suptitle_size,
                 xlabel,
                 xlabel_size,
                 ylabel,
                 ylabel_size,
                 tick_size):
        self._fig, self._ax = plt.subplots(1, 1, figsize=(15, 10), layout="tight")
        self._fig.suptitle(suptitle, fontsize=suptitle_size)
        self._ax.set_xlabel(xlabel, fontsize=xlabel_size)
        self._ax.set_ylabel(ylabel, fontsize=ylabel_size)
        self._ax.tick_params(axis='both', labelsize=tick_size)

    def get_fig(self):
        return self._fig

    def get_ax(self):
        return self._ax


if __name__ == "__main__":
    # setting our desired benchmark results table
    old_method_results_df = pd.DataFrame(columns=['BENCHMARK',
                                                  'MEAN_LIM2_FP_RATE', 'MEAN_LIM2_DETECTED_TROJ',
                                                  'MEAN_LIM4_FP_RATE', 'MEAN_LIM4_DETECTED_TROJ',
                                                  'MEAN_INF_FP_RATE', 'MEAN_INF_DETECTED_TROJ',
                                                  'MEDIAN_LIM2_FP_RATE', 'MEDIAN_LIM2_DETECTED_TROJ',
                                                  'MEDIAN_LIM4_FP_RATE', 'MEDIAN_LIM4_DETECTED_TROJ',
                                                  'MEDIAN_INF_FP_RATE', 'MEDIAN_INF_DETECTED_TROJ'])

    new_method_results_df = pd.DataFrame(columns=['BENCHMARK',
                                                  'MEAN_INF-LIM2_FP_RATE', 'MEAN_INF-LIM2_DETECTED_TROJ',
                                                  'MEAN_INF-LIM4_FP_RATE', 'MEAN_INF-LIM4_DETECTED_TROJ',
                                                  'MEDIAN_INF-LIM2_FP_RATE', 'MEDIAN_INF-LIM2_DETECTED_TROJ',
                                                  'MEDIAN_INF-LIM4_FP_RATE', 'MEDIAN_INF-LIM4_DETECTED_TROJ'])

    normalized_new_method_results_df = pd.DataFrame(columns=['BENCHMARK',
                                                             'MEAN_INF-LIM2_NORM_FP_RATE', 'MEAN_INF-LIM2_NORM_DETECTED_TROJ',
                                                             'MEAN_INF-LIM4_NORM_FP_RATE', 'MEAN_INF-LIM4_NORM_DETECTED_TROJ',
                                                             'MEDIAN_INF-LIM2_NORM_FP_RATE', 'MEDIAN_INF-LIM2_NORM_DETECTED_TROJ',
                                                             'MEDIAN_INF-LIM4_NORM_FP_RATE', 'MEDIAN_INF-LIM4_NORM_DETECTED_TROJ'])

    # setting up the threshold
    thresholds_for_old_method = [0.001, 0.025, 0.06]  # <=
    thresholds_for_new_method = [1]  # >=
    thresholds_for_normalized_new_method = [0.8, 0.9, 1]  # >=

    # defining x and y lists for 2 figures (each set will contain unique x,y tuples)
    inf_size_per_mean = set()
    inf_min_lim2_size_per_val = set()

    # Create a custom color palette with enough unique colors
    palette = sns.color_palette(cc.glasbey, n_colors=len(BENCHS_TO_CALC))
    print(len(palette))

    # for fig 4 in the paper
    fig_4 = Figure(
        "Mean of control-values vs. Fan-in of a gate output wire",
        24,
        'Fan-in',
        24,
        'Mean of control-value',
        24,
        20
    )

    # for fig 5 in the paper
    fig_5 = Figure(
        "b03 - Regional sensitivity vs. Logic depth",
        24,
        'Logic depth',
        24,
        'Regional sensitivity of a wire',
        24,
        20
    )

    # for fig 6 in the paper
    fig_6 = Figure(
        "Regional sensitivity vs. Logic depth",
        24,
        'Logic depth',
        24,
        'Regional sensitivity of a wire',
        24,
        20
    )

    # for fig 7 in the paper
    fig_7 = Figure(
        "Exponent fit power cofactor over linear fit cofactor",
        24,
        'Exponent fit power cofactor',
        24,
        'Linear fit cofactor',
        24,
        20
    )

    a_fit_inf_min_lim2_lst = []
    i = 0
    # go over completed benchmarks in output folder
    for entry in sorted(os.listdir(BENCHS_OUTPUT_FOLDER)):

        entry_path = os.path.join(BENCHS_OUTPUT_FOLDER, entry)
        if os.path.isdir(entry_path) and entry in BENCHS_TO_CALC:
            calc_df_old_fanci, calc_df_fanci_dist = handle_bench_calculation(entry_path)
            sorted_calc_df_old_fanci = calc_df_old_fanci.sort_values(by='INF_MEAN')
            sorted_calc_df_fanci_dist = calc_df_fanci_dist.sort_values(by='INF-LIM2_SIZE')

            # calculating the methods

            for t in thresholds_for_old_method:
                lim2_mean_fp_rate, lim2_mean_detected_trojan = calc_proportions_old_method(calc_df_old_fanci, t, entry + ".v", 'LIM2_MEAN')
                lim4_mean_fp_rate, lim4_mean_detected_trojan = calc_proportions_old_method(calc_df_old_fanci, t, entry + ".v", 'LIM4_MEAN')
                inf_mean_fp_rate, inf_mean_detected_trojan = calc_proportions_old_method(calc_df_old_fanci, t, entry + ".v", 'INF_MEAN')
                lim2_median_fp_rate, lim2_median_detected_trojan = calc_proportions_old_method(calc_df_old_fanci, t, entry + ".v", 'LIM2_MEDIAN')
                lim4_median_fp_rate, lim4_median_detected_trojan = calc_proportions_old_method(calc_df_old_fanci, t, entry + ".v", 'LIM4_MEDIAN')
                inf_median_fp_rate, inf_median_detected_trojan = calc_proportions_old_method(calc_df_old_fanci, t, entry + ".v", 'INF_MEDIAN')

                old_method_results_df = pd.concat([old_method_results_df, pd.DataFrame({'BENCHMARK': f'{entry}\n(for <= {t})',
                                                  'MEAN_LIM2_FP_RATE':lim2_mean_fp_rate, 'MEAN_LIM2_DETECTED_TROJ':lim2_mean_detected_trojan,
                                                  'MEAN_LIM4_FP_RATE':lim4_mean_fp_rate, 'MEAN_LIM4_DETECTED_TROJ':lim4_mean_detected_trojan,
                                                  'MEAN_INF_FP_RATE':inf_mean_fp_rate, 'MEAN_INF_DETECTED_TROJ':inf_mean_detected_trojan,
                                                  'MEDIAN_LIM2_FP_RATE':lim2_median_fp_rate, 'MEDIAN_LIM2_DETECTED_TROJ':lim2_median_detected_trojan,
                                                  'MEDIAN_LIM4_FP_RATE':lim4_median_fp_rate, 'MEDIAN_LIM4_DETECTED_TROJ':lim4_median_detected_trojan,
                                                  'MEDIAN_INF_FP_RATE':inf_median_fp_rate, 'MEDIAN_INF_DETECTED_TROJ':inf_median_detected_trojan},
                                                  index=[len(old_method_results_df)])], ignore_index=True)

            for t in thresholds_for_new_method:
                inf_min_lim2_mean_fp_rate, inf_min_lim2_mean_detected_trojan = calc_proportions_new_method(calc_df_fanci_dist, t, entry + ".v", 'INF-LIM2_MEAN')
                inf_min_lim4_mean_fp_rate, inf_min_lim4_mean_detected_trojan = calc_proportions_new_method(calc_df_fanci_dist, t, entry + ".v", 'INF-LIM4_MEAN')
                inf_min_lim2_median_fp_rate, inf_min_lim2_median_detected_trojan = calc_proportions_new_method(calc_df_fanci_dist, t, entry + ".v", 'INF-LIM2_MEDIAN')
                inf_min_lim4_median_fp_rate, inf_min_lim4_median_detected_trojan = calc_proportions_new_method(calc_df_fanci_dist, t, entry + ".v", 'INF-LIM4_MEDIAN')

                new_method_results_df = pd.concat(
                    [new_method_results_df, pd.DataFrame({'BENCHMARK': f'{entry}\n(for >= {t})',
                                                          'MEAN_INF-LIM2_FP_RATE': inf_min_lim2_mean_fp_rate,
                                                          'MEAN_INF-LIM2_DETECTED_TROJ': inf_min_lim2_mean_detected_trojan,
                                                          'MEAN_INF-LIM4_FP_RATE': inf_min_lim4_mean_fp_rate,
                                                          'MEAN_INF-LIM4_DETECTED_TROJ': inf_min_lim4_mean_detected_trojan,
                                                          'MEDIAN_INF-LIM2_FP_RATE': inf_min_lim2_median_fp_rate,
                                                          'MEDIAN_INF-LIM2_DETECTED_TROJ': inf_min_lim2_median_detected_trojan,
                                                          'MEDIAN_INF-LIM4_FP_RATE': inf_min_lim4_median_fp_rate,
                                                          'MEDIAN_INF-LIM4_DETECTED_TROJ': inf_min_lim4_median_detected_trojan},
                                                         index=[len(old_method_results_df)])], ignore_index=True)
            for t in thresholds_for_normalized_new_method:
                inf_min_lim2_norm_mean_fp_rate, inf_min_lim2_norm_mean_detected_trojan = calc_proportions_new_method(calc_df_fanci_dist, t, entry + ".v", 'INF-LIM2_NORM_MEAN')
                inf_min_lim4_norm_mean_fp_rate, inf_min_lim4_norm_mean_detected_trojan = calc_proportions_new_method(calc_df_fanci_dist, t, entry + ".v", 'INF-LIM4_NORM_MEAN')
                inf_min_lim2_norm_median_fp_rate, inf_min_lim2_norm_median_detected_trojan = calc_proportions_new_method(calc_df_fanci_dist, t, entry + ".v", 'INF-LIM2_NORM_MEDIAN')
                inf_min_lim4_norm_median_fp_rate, inf_min_lim4_norm_median_detected_trojan = calc_proportions_new_method(calc_df_fanci_dist, t, entry + ".v", 'INF-LIM4_NORM_MEDIAN')

                normalized_new_method_results_df = pd.concat(
                    [normalized_new_method_results_df, pd.DataFrame({'BENCHMARK': f'{entry}\n(for >= {t})',
                                                          'MEAN_INF-LIM2_NORM_FP_RATE': inf_min_lim2_norm_mean_fp_rate,
                                                          'MEAN_INF-LIM2_NORM_DETECTED_TROJ': inf_min_lim2_norm_mean_detected_trojan,
                                                          'MEAN_INF-LIM4_NORM_FP_RATE': inf_min_lim4_norm_mean_fp_rate,
                                                          'MEAN_INF-LIM4_NORM_DETECTED_TROJ': inf_min_lim4_norm_mean_detected_trojan,
                                                          'MEDIAN_INF-LIM2_NORM_FP_RATE': inf_min_lim2_norm_median_fp_rate,
                                                          'MEDIAN_INF-LIM2_NORM_DETECTED_TROJ': inf_min_lim2_norm_median_detected_trojan,
                                                          'MEDIAN_INF-LIM4_NORM_FP_RATE': inf_min_lim4_norm_median_fp_rate,
                                                          'MEDIAN_INF-LIM4_NORM_DETECTED_TROJ': inf_min_lim4_norm_median_detected_trojan},
                                                         index=[len(old_method_results_df)])], ignore_index=True)

            inf_size_per_mean = set((x, y) for x, y in zip(list(sorted_calc_df_old_fanci['INF_SIZE'].values), list(sorted_calc_df_old_fanci['INF_MEAN'].values)))
            inf_min_lim2_size_per_val = set((x, y) for x, y in zip(list(calc_df_fanci_dist['INF-LIM2_SIZE'].values), list(calc_df_fanci_dist['INF-LIM2_NORM_MEAN'].values)) if y != np.inf and y != np.nan)

            # sorts the sets
            inf_size_per_mean = sorted(inf_size_per_mean, key=lambda tup: tup[0])
            inf_min_lim2_size_per_val = sorted(inf_min_lim2_size_per_val, key=lambda tup: tup[0])

            # gets x and y values
            inf_size_per_mean_x = [tup[0] for tup in inf_size_per_mean]
            inf_size_per_mean_y = [tup[1] for tup in inf_size_per_mean]
            inf_min_lim2_size_per_val_x = [tup[0] for tup in inf_min_lim2_size_per_val]
            inf_min_lim2_size_per_val_y = [tup[1] for tup in inf_min_lim2_size_per_val]

            # Perform the curve fit
            params_inf, covariance_inf = curve_fit(model_function_exp, inf_size_per_mean_x, inf_size_per_mean_y, maxfev=8000)
            if len(inf_min_lim2_size_per_val_x) > 1 and len(inf_min_lim2_size_per_val_y) > 1:
                params_inf_min_lim2, covariance_inf_min_lim2 = curve_fit(model_function_lin, inf_min_lim2_size_per_val_x, inf_min_lim2_size_per_val_y, maxfev=8000)

                # Get the fitted parameters
                a_fit_inf, b_fit_inf, c_fit_inf = params_inf
                a_fit_inf_min_lim2, b_fit_inf_min_lim2 = params_inf_min_lim2

                # Generate y values for the fitted curve
                y_fit_inf = model_function_exp(inf_size_per_mean_x, a_fit_inf, b_fit_inf, c_fit_inf)
                y_fit_inf_min_lim2 = model_function_lin(inf_min_lim2_size_per_val_x, a_fit_inf_min_lim2, b_fit_inf_min_lim2)

                # performing scatter in the results

                # fig 4
                fig_4.get_ax().scatter(inf_size_per_mean_x, inf_size_per_mean_y, color=palette[i])
                fig_4.get_ax().plot(inf_size_per_mean_x, y_fit_inf, linestyle='--', color=palette[i])

                # fig 5
                if entry == "b03":
                    fig_5.get_ax().scatter(inf_min_lim2_size_per_val_x, inf_min_lim2_size_per_val_y, color=palette[i])
                    fig_5.get_ax().plot(inf_min_lim2_size_per_val_x, y_fit_inf_min_lim2, linestyle='--', color=palette[i])

                # fig 6
                fig_6.get_ax().scatter(inf_min_lim2_size_per_val_x, inf_min_lim2_size_per_val_y, color=palette[i])
                fig_6.get_ax().plot(inf_min_lim2_size_per_val_x, y_fit_inf_min_lim2, linestyle='--', color=palette[i])

                # fig 7
                a_fit_inf_min_lim2_lst.append(a_fit_inf_min_lim2)
                fig_7.get_ax().scatter(b_fit_inf, a_fit_inf_min_lim2, color=palette[i])

                # info on the entries
                rgb = palette[i]
                hex_color = f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'
                print(f"{entry} has color:{hex_color}")
                i += 1

    # fig 6 again
    fig_6.get_ax().set_ylim(-1.7, 2.5)

    # fig 7 again
    from statistics import mean
    fig_7.get_ax().axhline(mean(a_fit_inf_min_lim2_lst), linestyle='--', color='red')

    # Display the plot
    plt.show()

    # saving all dfs
    old_method_results_df.to_csv(os.path.join(CALC_OUTPUT_FOLDER, 'old_method_results_df.csv'))
    new_method_results_df.to_csv(os.path.join(CALC_OUTPUT_FOLDER, 'new_method_results_df.csv'))
    normalized_new_method_results_df.to_csv(os.path.join(CALC_OUTPUT_FOLDER, 'normalized_new_method_results_df.csv'))

    # Convert CSV to LaTeX
    df_to_latex(old_method_results_df, 'output_table.tex')
    print(f"Conversion complete. LaTeX table saved to output_table.tex.")








