import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import re
import argparse

def process_netlist_name_and_limit(df_path):
    # Get the parent directory of the given file path
    parent_directory = os.path.dirname(df_path)

    # Extract the last component (immediate parent directory) from the path
    immediate_parent_directory = os.path.basename(parent_directory)

    #extracts the limit
    limit = 0
    df_basename = os.path.splitext(os.path.basename(df_path))[0]
    if "_inf_" in df_basename:
        limit = -1
    else:
        # Define the pattern using a regular expression
        pattern = r"_lim(\d+)_"

        # Use re.search to find the pattern in the input string
        match = re.search(pattern, df_basename)

        if match:
            # The pattern is found, extract the number from the match
            limit = int(match.group(1))

    return immediate_parent_directory, limit

if __name__ == "__main__":
    # parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('df1_path', type=str, help='path of df1')
    parser.add_argument('df2_path', type=str, help='path of df2')
    args = parser.parse_args()

    df1_netlist_name, df1_limit = process_netlist_name_and_limit(args.df1_path)
    df2_netlist_name, df2_limit = process_netlist_name_and_limit(args.df2_path)

    df1 = pd.read_csv(args.df1_path)
    df2 = pd.read_csv(args.df2_path)
    diff_column = None

    if (df1_limit >= df2_limit and df2_limit != -1) or df1_limit == -1:
        diff_column = abs(df2['MEAN'] - df1['MEAN']) / df1['MEAN']
        #diff_column2 = (df1['MEDIAN'] - df2['MEDIAN']) / df1['MEDIAN']
    else:
        diff_column = (df1['MEAN'] - df2['MEAN']) / df2['MEAN']
        #diff_column2 = (df2['MEDIAN'] - df1['MEDIAN']) / df2['MEDIAN']

    diff_column3 = abs(df2['SIZE'] - df1['SIZE'])

    new_df = pd.DataFrame({ 'GATE_NAME': df1['GATE_NAME'], 'GATE_TYPE': df1['GATE_TYPE'],
                            'MEAN_DIFF': diff_column, 'SIZE_DIFF':diff_column3}).sort_values(by='MEAN_DIFF', ascending=False)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(new_df)





