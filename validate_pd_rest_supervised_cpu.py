#!/usr/bin/env python3
# ============================================================
# FINAL SUPERVISED EXTERNAL VALIDATION (PD REST)
# Excel labels + EEG + Subject-level Metrics
# ============================================================

# ------------------------------------------------------------
# CPU SAFETY (PREVENT SYSTEM FREEZE)
# ------------------------------------------------------------
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import re
import csv
import collections
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.io as sio
import mne
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix
)

# ------------------------------------------------------------
# MODEL (NO TRAINING SIDE EFFECTS)
# ------------------------------------------------------------
from models.neuroconvformer import NeuroConvFormerLite

# ============================================================
# USER PATHS (EDIT IF REQUIRED)
# ============================================================
PD_REST_DIR = r"C:\projects\NCF-lite\dataset\PD REST"
MODEL_PATH  = r"C:\projects\NCF-lite\Models\neuroconvformer_lite_mil_ds004584_best.pt"

EXCEL_FILES = [
    r"C:\projects\NCF-lite\dataset\PD REST\IMPORT_ME_REST.xlsx",
    r"C:\projects\NCF-lite\dataset\PD REST\IMPORT_ME_REST1.xlsx"
]

CSV_OUT = "pd_rest_supervised_validation.csv"

DEVICE = "cpu"

WIN_SEC = 10.0
OVERLAP = 0.5
TRAIN_CHANNELS = 60
BATCH_SIZE = 8
MAX_WINDOWS = 200   # laptop-safe; remove for long overnight runs

# ============================================================
# STEP 1: LOAD LABELS FROM EXCEL (PD_ID / MATCHCTL_ID)
# ============================================================
label_map = {}

for xf in EXCEL_FILES:
    df = pd.read_excel(xf)

    for col in df.columns:
        cname = col.strip().upper()

        if cname == "PD_ID":
            for v in df[col].dropna():
                label_map[int(v)] = 1   # PD

        elif cname == "MATCHCTL_ID":
            for v in df[col].dropna():
                label_map[int(v)] = 0   # Healthy Control

print(f"[INFO] Loaded labels for {len(label_map)} subjects from Excel")

# ============================================================
# SUBJECT ID FROM EEG FILENAME
# ============================================================
def infer_subject_id(fname):
    m = re.match(r"(\d+)_", fname)
    if not m:
        return None
    return int(m.group(1))

# ============================================================
# LOAD EEGLAB .MAT → MNE RAW
# ============================================================
def load_mat_raw(fp):
    mat = sio.loadmat(fp, squeeze_me=True, struct_as_record=False)

    if "EEG" not in mat:
        raise KeyError("EEG struct not found in MAT file")

    EEG = mat["EEG"]
    data = EEG.data

    if data.ndim == 3:
        data = data[0]
    if data.shape[0] > data.shape[1]:
        data = data.T

    sfreq = float(EEG.srate)
    ch_names = [cl.labels.strip() for cl in EEG.chanlocs]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data.astype(np.float32), info, verbose=False)

# ============================================================
# WINDOW EXTRACTION
# ============================================================
def extract_windows(raw, picks):
    data = raw.get_data(picks=picks)
    sfreq = raw.info["sfreq"]

    win_len = int(WIN_SEC * sfreq)
    hop = int(win_len * (1 - OVERLAP))

    if data.shape[1] < win_len:
        return None

    windows = []
    for s in range(0, data.shape[1] - win_len + 1, hop):
        windows.append(data[:, s:s + win_len])

    return np.stack(windows).astype(np.float32)

# ============================================================
# BATCHED INFERENCE (CPU-SAFE)
# ============================================================
def batched_inference(model, X, batch_size):
    probs = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size]
        with torch.no_grad():
            p = torch.softmax(model(xb), dim=1)[:, 1]
        probs.append(p.cpu())
    return torch.cat(probs)

# ============================================================
# DISCOVER CANONICAL EEG CHANNELS
# ============================================================
files = glob(os.path.join(PD_REST_DIR, "*.mat"))
channel_sets = []

for fp in files:
    raw = load_mat_raw(fp)
    channel_sets.append(set(c.upper() for c in raw.ch_names))

CANON_CHS = sorted(set.intersection(*channel_sets))[:TRAIN_CHANNELS]

def get_picks(raw):
    idx = {c.upper(): i for i, c in enumerate(raw.ch_names)}
    return [idx[c] for c in CANON_CHS]

print(f"[INFO] Using {len(CANON_CHS)} canonical EEG channels")

# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================
model = NeuroConvFormerLite(n_ch=TRAIN_CHANNELS).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

print("[INFO] Model loaded (CPU mode)")

# ============================================================
# STEP 2: SUBJECT-LEVEL INFERENCE
# ============================================================
subject_probs = defaultdict(list)
subject_labels = {}

print("[INFO] Running supervised external validation...")

for fp in tqdm(files, desc="Validating EEG files"):
    sid = infer_subject_id(os.path.basename(fp))

    if sid is None or sid not in label_map:
        continue

    raw = load_mat_raw(fp)
    picks = get_picks(raw)
    Xw = extract_windows(raw, picks)

    if Xw is None:
        continue

    if Xw.shape[0] > MAX_WINDOWS:
        Xw = Xw[:MAX_WINDOWS]

    X = torch.from_numpy(Xw).to(DEVICE)

    # same normalization as training
    X = (X - X.mean(dim=2, keepdim=True)) / (X.std(dim=2, keepdim=True) + 1e-6)

    probs = batched_inference(model, X, BATCH_SIZE)

    subject_probs[sid].append(probs.numpy())
    subject_labels[sid] = label_map[sid]

# ============================================================
# STEP 3: METRICS (SAFE)
# ============================================================
y_true = []
y_score = []

for sid in sorted(subject_probs):
    mean_prob = np.concatenate(subject_probs[sid]).mean()
    y_score.append(mean_prob)
    y_true.append(subject_labels[sid])

# DEBUG: CLASS DISTRIBUTION
print("[DEBUG] Label distribution:", collections.Counter(y_true))

y_true = np.array(y_true)
y_score = np.array(y_score)
y_pred = (y_score >= 0.5).astype(int)

acc = accuracy_score(y_true, y_pred)
bacc = balanced_accuracy_score(y_true, y_pred)

if len(set(y_true)) == 2:
    auc = roc_auc_score(y_true, y_score)
else:
    auc = None

cm = confusion_matrix(y_true, y_pred)

# ============================================================
# SAVE SUBJECT-LEVEL RESULTS
# ============================================================
with open(CSV_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject_id", "true_label", "mean_pd_probability"])
    for sid in sorted(subject_probs):
        writer.writerow([sid, subject_labels[sid],
                         np.concatenate(subject_probs[sid]).mean()])

# ============================================================
# PRINT FINAL RESULTS
# ============================================================
print("\n=== FINAL SUPERVISED EXTERNAL VALIDATION ===")
print(f"Accuracy           : {acc:.4f}")
print(f"Balanced Accuracy  : {bacc:.4f}")
print(f"ROC-AUC            : {auc if auc is not None else 'Not defined (single class)'}")
print("Confusion Matrix:")
print(cm)
print(f"\n[INFO] Results saved to {CSV_OUT}")
