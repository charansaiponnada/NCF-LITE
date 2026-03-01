# from pathlib import Path
# import pandas as pd

# def makeExcel():
#     base_dir = Path(__file__).resolve().parent.parent
#     data_dir = base_dir / "dataset" / "PD REST"

#     files = [
#         data_dir / "IMPORT_ME_REST.xlsx",
#         data_dir / "IMPORT_ME_REST1.xlsx"
#     ]

#     for file in files:
#         print("Reading:", file)
#         df_dict = pd.read_excel(file, sheet_name=None)

#         for sheet, df in df_dict.items():
#             print(sheet)
#             print(df.head())

# makeExcel()


import os
import pandas as pd
import re

EXCEL1 = r"C:\projects\NCF-lite\dataset\PD REST\IMPORT_ME_REST.xlsx"
EXCEL2 = r"C:\projects\NCF-lite\dataset\PD REST\IMPORT_ME_REST1.xlsx"

EEG_DIR = r"C:\projects\NCF-lite\dataset\PD REST"

# 1. Load both Excel files
df1 = pd.read_excel(EXCEL1)
df2 = pd.read_excel(EXCEL2)

# 2. Collect PD and Control IDs
pd_ids = set(df1["PD_ID"].dropna().astype(int)) | set(df2["PD_ID"].dropna().astype(int))
hc_ids = set(df1["MATCH CTL_ID"].dropna().astype(int)) | set(df2["MATCH CTL_ID"].dropna().astype(int))

print("PD subjects:", sorted(pd_ids))
print("HC subjects:", sorted(hc_ids))


# PD subjects: [801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829]
# HC subjects: [890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 8010, 8060, 8070]

# 3. Scan EEG directory for .mat files
rows = []

for fname in os.listdir(EEG_DIR):
    if not fname.endswith(".mat"):
        continue

    # Extract subject ID from filename
    m = re.match(r"(\d+)_", fname)
    if not m:
        continue

    sid = int(m.group(1))

    if sid in pd_ids:
        label = "PD"
    elif sid in hc_ids:
        label = "Healthy"
    else:
        continue   # ignore files not listed in Excel

    rows.append([fname, label])

# 4. Create dataframe
df_sessions = pd.DataFrame(rows, columns=["eeg_session_file", "class_label"])

print(df_sessions.head())
print("\nTotal sessions:", len(df_sessions))
print(df_sessions["class_label"].value_counts())

# 5. Save dataset
out_path = os.path.join(r"C:\projects\NCF-lite\test-2", "eeg_sessions_labels.xlsx")
df_sessions.to_excel(out_path, index=False)

print("\nSaved:", out_path)
