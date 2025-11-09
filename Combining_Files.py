import os 
import pandas as pd
import sys
print(sys.executable)
folder = "/Users/yogeshsharma/Desktop/Intrusion_Detection_Project/Data/archive (1)"
csv_files = []

for f in os.listdir(folder):
    if f.endswith(".csv"):
        csv_files.append(f)

# for f in csv_files:
#     print(f)

# collection of dataframes
dfs = []

for f in csv_files:
    file_path = os.path.join(folder,f)
    df = pd.read_csv(file_path)
    dfs.append(df)

combined_dfs = pd.concat(dfs,ignore_index=True)
# combined_df = pd.concat(f for f in csv_files)

output_path = "/Users/yogeshsharma/Desktop/Intrusion_Detection_Project/Data/Combined_data/Combined_data.csv"
combined_dfs.to_csv(output_path,index=False)