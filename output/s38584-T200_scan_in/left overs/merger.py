import pandas as pd

old_df = pd.read_csv("s38584-T200_scan_in_inf_20231127-224906_test.csv")
new_df = pd.read_csv("s38584-T200_scan_in_inf_20231229-134903_test.csv")

merged_df = pd.concat([old_df.iloc[0:len(old_df)-1], new_df], ignore_index=True)
merged_df.to_csv("s38584-T200_scan_in_inf_merged_test.csv")
