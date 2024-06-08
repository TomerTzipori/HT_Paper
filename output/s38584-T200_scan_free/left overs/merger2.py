import pandas as pd

old_df = pd.read_csv("s38584-T200_scan_free_inf_20231223-233454_test.csv")
new_df = pd.read_csv("s38584-T200_scan_free_inf_20231229-140438_test.csv")

merged_df = pd.concat([old_df, new_df.iloc[913:]], ignore_index=True)
merged_df.to_csv("s38584-T200_scan_free_inf_merged_test.csv")
