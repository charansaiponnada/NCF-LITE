import pandas as pd
import os

xlsx_path = r"C:\projects\NCF-lite\test-2\eeg_sessions_labels.xlsx"

df = pd.read_excel(xlsx_path)

df["eeg_session_file"] = df["eeg_session_file"].str.replace(".mat", ".edf", regex=False)

df.to_excel(xlsx_path, index=False)

print("Updated Excel to use .edf files")
