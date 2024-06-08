import pandas as pd

old_df = pd.read_csv("s38584-T100_scan_in_inf_20231223-233221_test.csv")
new_df = pd.read_csv("s38584-T100_scan_in_inf_20231229-140403_test.csv")

merged_df = pd.concat([old_df.iloc[0:2139], new_df.iloc[913:]], ignore_index=True)
merged_df.to_csv("s38584-T100_scan_in_inf_merged_test.csv")
