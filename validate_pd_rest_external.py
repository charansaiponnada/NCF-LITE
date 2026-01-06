#!/usr/bin/env python3
# ============================================================
# External Validation on PD REST Dataset
# Pretrained model: ds004584 (best fold)
# Robust, side-effect free, research-correct
# ============================================================

import os
import re
from glob import glob
import numpy as np
import scipy.io as sio
from tqdm import tqdm

import torch
import mne

# ------------------------------------------------------------
# IMPORT MODEL ONLY (NO TRAINING SIDE EFFECTS)
# ------------------------------------------------------------
from models.neuroconvformer import NeuroConvFormerLite

# ============================================================
# USER PATHS
# ============================================================
PD_REST_DIR = r"C:\projects\NCF-lite\dataset\PD REST"
MODEL_PATH  = r"C:\projects\NCF-lite\Models\neuroconvformer_lite_mil_ds004584_best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WIN_SEC = 10.0
OVERLAP = 0.5
TRAIN_CHANNELS = 60   # IMPORTANT: ds004584 trained on 60 channels

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
        raise KeyError("EEG struct not found in MAT file")

    EEG = mat["EEG"]

    data = EEG.data
    if data.ndim == 3:
        data = data[0]

    if data.shape[0] > data.shape[1]:
        data = data.T

    sfreq = float(EEG.srate)

    if hasattr(EEG, "chanlocs"):
        ch_names = [cl.labels.strip() for cl in EEG.chanlocs]
    else:
        ch_names = [f"Ch{i}" for i in range(data.shape[0])]

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
# DISCOVER CANONICAL CHANNELS (PD REST)
# ============================================================
files = glob(os.path.join(PD_REST_DIR, "*.mat"))
all_channel_sets = []

print("[INFO] Scanning PD REST dataset...")
for fp in files:
    raw = load_mat_raw(fp)
    all_channel_sets.append(set(c.upper() for c in raw.ch_names))

CANON_CHS = sorted(set.intersection(*all_channel_sets))
print(f"[INFO] Canonical EEG channels found: {len(CANON_CHS)}")

# ------------------------------------------------------------
# FORCE CHANNEL COUNT TO MATCH PRETRAINED MODEL
# ------------------------------------------------------------
if len(CANON_CHS) < TRAIN_CHANNELS:
    raise RuntimeError(
        f"Only {len(CANON_CHS)} channels found, "
        f"but pretrained model expects {TRAIN_CHANNELS}"
    )

CANON_CHS = CANON_CHS[:TRAIN_CHANNELS]
print(f"[INFO] Using first {TRAIN_CHANNELS} canonical channels")

def get_picks(raw):
    name_to_idx = {c.upper(): i for i, c in enumerate(raw.ch_names)}
    return [name_to_idx[c] for c in CANON_CHS]

# ============================================================
# LOAD EEG WINDOWS
# ============================================================
subjects = {}

print("[INFO] Loading EEG windows...")
for fp in tqdm(files):
    sid = infer_subject_id(os.path.basename(fp))
    raw = load_mat_raw(fp)
    picks = get_picks(raw)

    Xw = extract_windows(raw, picks)
    if Xw is None:
        continue

    subjects.setdefault(sid, []).append(Xw)

# ============================================================
# LOAD PRETRAINED MODEL (NO RETRAINING)
# ============================================================
model = NeuroConvFormerLite(n_ch=TRAIN_CHANNELS).to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

# IMPORTANT: strict=False fixes head mismatch safely
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

print(f"[INFO] Loaded pretrained model: {MODEL_PATH}")

# ============================================================
# SUBJECT-LEVEL INFERENCE
# ============================================================
print("\nSubject | Mean PD Prob | Std | Windows")
print("-" * 45)

with torch.no_grad():
    for sid in sorted(subjects):
        X = np.concatenate(subjects[sid], axis=0)
        X = torch.from_numpy(X).to(DEVICE)

        # SAME normalization as training
        X = (X - X.mean(dim=2, keepdim=True)) / (X.std(dim=2, keepdim=True) + 1e-6)

        probs = torch.softmax(model(X), dim=1)[:, 1]

        print(
            f"{sid:>6} | {probs.mean():.4f} | {probs.std():.4f} | {len(probs)}"
        )

print("\n[INFO] External validation completed successfully.")
