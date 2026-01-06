#!/usr/bin/env python3
# ============================================================
# CPU-OPTIMIZED External Validation on PD REST Dataset
# Subject-level aggregation + CSV logging
# ============================================================

# ------------------------------------------------------------
# CRITICAL: LIMIT CPU THREADS (PREVENT UI FREEZE)
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
from glob import glob
from collections import defaultdict

import numpy as np
import scipy.io as sio
from tqdm import tqdm
import mne

# ------------------------------------------------------------
# Import MODEL ONLY (NO TRAINING SIDE EFFECTS)
# ------------------------------------------------------------
from models.neuroconvformer import NeuroConvFormerLite

# ============================================================
# USER PATHS
# ============================================================
PD_REST_DIR = r"C:\projects\NCF-lite\dataset\PD REST"
MODEL_PATH  = r"C:\projects\NCF-lite\Models\neuroconvformer_lite_mil_ds004584_best.pt"

DEVICE = "cpu"   # FORCE CPU (Intel Iris laptop)

WIN_SEC = 10.0
OVERLAP = 0.5
TRAIN_CHANNELS = 60     # MUST match ds004584 training
BATCH_SIZE = 8          # CPU-safe
MAX_WINDOWS = 200       # laptop-safe (remove for overnight runs)

CSV_OUT = "pd_rest_external_validation.csv"

# ============================================================
# SUBJECT ID PARSER
# ============================================================
def infer_subject_id(fname):
    m = re.match(r"(\d+)_", fname)
    if not m:
        raise ValueError(f"Cannot parse subject ID from {fname}")
    return m.group(1)

# ============================================================
# LOAD EEGLAB .MAT EEG → MNE RAW
# ============================================================
def load_mat_raw(fp):
    mat = sio.loadmat(fp, squeeze_me=True, struct_as_record=False)

    if "EEG" not in mat:
        raise KeyError("EEG struct not found")

    EEG = mat["EEG"]

    data = EEG.data
    if data.ndim == 3:
        data = data[0]

    if data.shape[0] > data.shape[1]:
        data = data.T

    sfreq = float(EEG.srate)

    ch_names = (
        [cl.labels.strip() for cl in EEG.chanlocs]
        if hasattr(EEG, "chanlocs")
        else [f"Ch{i}" for i in range(data.shape[0])]
    )

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types="eeg"
    )

    return mne.io.RawArray(data.astype(np.float32), info, verbose=False)

# ============================================================
# WINDOW EXTRACTION
# ============================================================
def extract_windows(raw, picks):
    data = raw.get_data(picks=picks)
    sfreq = raw.info["sfreq"]

    win_len = int(WIN_SEC * sfreq)
    hop = max(1, int(win_len * (1 - OVERLAP)))

    if data.shape[1] < win_len:
        return None

    windows = []
    for s in range(0, data.shape[1] - win_len + 1, hop):
        windows.append(data[:, s:s + win_len])

    return np.stack(windows).astype(np.float32)

# ============================================================
# BATCHED INFERENCE (CRITICAL FOR CPU)
# ============================================================
def batched_pd_inference(model, X, batch_size):
    probs_all = []

    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size]
        with torch.no_grad():
            p = torch.softmax(model(xb), dim=1)[:, 1]
        probs_all.append(p.cpu())

    return torch.cat(probs_all)

# ============================================================
# DISCOVER CANONICAL EEG CHANNELS (PD REST)
# ============================================================
files = glob(os.path.join(PD_REST_DIR, "*.mat"))
all_channel_sets = []

print("[INFO] Scanning PD REST dataset...")
for fp in files:
    raw = load_mat_raw(fp)
    all_channel_sets.append(set(c.upper() for c in raw.ch_names))

CANON_CHS = sorted(set.intersection(*all_channel_sets))
print(f"[INFO] Canonical EEG channels found: {len(CANON_CHS)}")

if len(CANON_CHS) < TRAIN_CHANNELS:
    raise RuntimeError("Not enough channels to match pretrained model")

CANON_CHS = CANON_CHS[:TRAIN_CHANNELS]
print(f"[INFO] Using first {TRAIN_CHANNELS} canonical channels")

def get_picks(raw):
    name_to_idx = {c.upper(): i for i, c in enumerate(raw.ch_names)}
    return [name_to_idx[c] for c in CANON_CHS]

# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================
model = NeuroConvFormerLite(n_ch=TRAIN_CHANNELS).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

print("[INFO] Pretrained model loaded (CPU mode)")

# ============================================================
# STEP A: SUBJECT-LEVEL AGGREGATION
# ============================================================
subject_probs = defaultdict(list)
subject_windows = defaultdict(int)

print("\n[INFO] Running external validation (subject-level)...")

for fp in tqdm(files, desc="Validating recordings"):
    sid = infer_subject_id(os.path.basename(fp))

    raw = load_mat_raw(fp)
    picks = get_picks(raw)
    Xw = extract_windows(raw, picks)

    if Xw is None:
        continue

    if Xw.shape[0] > MAX_WINDOWS:
        Xw = Xw[:MAX_WINDOWS]

    X = torch.from_numpy(Xw).to(DEVICE)

    # SAME normalization as training
    X = (X - X.mean(dim=2, keepdim=True)) / (X.std(dim=2, keepdim=True) + 1e-6)

    probs = batched_pd_inference(model, X, BATCH_SIZE)

    subject_probs[sid].append(probs.numpy())
    subject_windows[sid] += len(probs)

# ============================================================
# STEP B: FINAL SUBJECT METRICS + CSV LOGGING
# ============================================================
results = []

print("\nSubject | Mean PD Prob | Std | Total Windows")
print("-" * 55)

for sid in sorted(subject_probs.keys()):
    all_probs = np.concatenate(subject_probs[sid])
    mean_p = all_probs.mean()
    std_p  = all_probs.std()

    print(f"{sid:>6} | {mean_p:.4f} | {std_p:.4f} | {subject_windows[sid]}")

    results.append([
        sid,
        float(mean_p),
        float(std_p),
        subject_windows[sid]
    ])

with open(CSV_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "subject_id",
        "mean_pd_probability",
        "std_pd_probability",
        "num_windows"
    ])
    writer.writerows(results)

print(f"\n[INFO] Results saved to: {CSV_OUT}")
print("[INFO] External validation COMPLETED successfully.")
