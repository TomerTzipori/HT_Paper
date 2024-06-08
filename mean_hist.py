import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import re
import argparse

def process_netlist_name_and_limit(args):
    # Get the parent directory of the given file path
    parent_directory = os.path.dirname(args.df_path)

    # Extract the last component (immediate parent directory) from the path
    immediate_parent_directory = os.path.basename(parent_directory)

    #extracts the limit
    limit = 0
    df_basename = os.path.splitext(os.path.basename(args.df_path))[0]
    if "_inf_" in df_basename:
        limit = -1
    else:
        # Define the pattern using a regular expression
        pattern = r"_lim(\d+)_"

        # Use re.search to find the pattern in the input string
        match = re.search(pattern, df_basename)

        if match:
            # The pattern is found, extract the number from the match
            limit = match.group(1)
            print(f"Substring found! Number: {limit}")

    return immediate_parent_directory, limit

if __name__ == "__main__":
    # parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('df_path', type=str, help='path of ')
    args = parser.parse_args()

    netlist_name, limit = process_netlist_name_and_limit(args)

    df = pd.read_csv(args.df_path)

    fig, ax = plt.subplots(1, 1, layout="constrained")
    ax.set_title("mean of: " + netlist_name + " with limit:" + str(limit))
    ax.hist(df['MEAN'], bins=1000, alpha=1)
    plt.show()
