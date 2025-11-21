import pandas as pd
from pathlib import Path

csv_out = Path('datasets/c-mapss/processed_data/engine_knee_plots_multi_no_normal/all_engines_labeled.csv')
df_all = pd.read_csv(csv_out)

print(df_all['state'].unique())
print(df_all['state'].min(), df_all['state'].max())
