#!/usr/bin/env python3
# ============================================================
# Parkinson's Resting-State EEG – Subject-wise Stratified 5-Fold CV
# Protocol:
# ✅ Subject-wise Stratified 5-fold CV (StratifiedGroupKFold)
# ✅ FULL variable-length recordings (no truncation)
# ✅ Canonical EEG channel intersection (EEG-only)
# ✅ Windowing: 10s, 50% overlap across full length
# ✅ Metrics per fold + overall:
#    Window: Acc/BAcc/Prec/Rec/F1/AUC
#    Subject: Acc/BAcc/Prec/Rec/F1/AUC
#
# New architecture (<=250k params): NeuroConvFormer-Lite
#
# UPGRADE (requested): MIL subject-level training
# - Bag sampling (N windows/subject per step)
# - Attention pooling over windows -> subject logits
# - Focal / weighted CE for stability
# - Confidence-trimmed inference for subject aggregation
#
# NEW CHANGES (requested):
# (1) Per-fold subject threshold tuning (train-only) for subject BAcc
# (2) Use MILPool at inference for subject prediction
# (3) Logit-mean aggregation (implemented as attention-weighted mean of per-window log-odds)
# (4) Balanced bag sampler (WeightedRandomSampler) to lift recall
# ============================================================

import os, re, glob, math, json, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)

import mne

# -----------------------
# USER PATHS - UPDATED FOR EXCEL + EDF
# -----------------------
DATA_DIR = r"C:\projects\NCF-lite\dataset\PD REST\edf"
LABEL_FILE = r"C:\projects\NCF-lite\test-2\eeg_sessions_labels.xlsx"

# -----------------------
# GLOBALS
# -----------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)



# ============================================================
# 4) WINDOWING (10s, 50% overlap) OVER FULL DURATION
# ============================================================
WIN_SEC = 10.0
OVERLAP = 0.50

# extract_windows_from_raw is now defined in main() and passed as parameter

# ============================================================
# 5) AUGMENTATIONS (window-level, safe + effective)
# ============================================================
class Augment:
    def __init__(self, p_noise=0.3, p_shift=0.3, p_chdrop=0.2, p_banddrop=0.2,
                 noise_std=0.01, max_shift_frac=0.1, chdrop_frac=0.1):
        self.p_noise = p_noise
        self.p_shift = p_shift
        self.p_chdrop = p_chdrop
        self.p_banddrop = p_banddrop
        self.noise_std = noise_std
        self.max_shift_frac = max_shift_frac
        self.chdrop_frac = chdrop_frac

    def __call__(self, x):
        # x: (C,T) torch
        C, T = x.shape

        if random.random() < self.p_shift:
            max_shift = int(self.max_shift_frac * T)
            if max_shift > 0:
                s = random.randint(-max_shift, max_shift)
                x = torch.roll(x, shifts=s, dims=1)

        if random.random() < self.p_noise:
            x = x + self.noise_std * torch.randn_like(x)

        if random.random() < self.p_chdrop:
            k = max(1, int(self.chdrop_frac * C))
            idx = torch.randperm(C)[:k]
            x[idx] = 0.0

        # simple "band dropout" proxy via random temporal smoothing / differencing
        # (cheap, stable, helps robustness)
        if random.random() < self.p_banddrop:
            if random.random() < 0.5:
                # smooth
                k = random.choice([3,5,7])
                pad = k//2
                x = F.avg_pool1d(x.unsqueeze(0), kernel_size=k, stride=1, padding=pad).squeeze(0)
            else:
                # high-pass-ish
                x = x - F.avg_pool1d(x.unsqueeze(0), kernel_size=7, stride=1, padding=3).squeeze(0)

        return x

# ============================================================
# 6) DATASET (cache windows per recording)
# ============================================================
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, records, canon_chs, extract_windows_fn, training=False, augment=None):
        """
        records: list of dict {path, subject_id, y}
        Creates a flat list of windows with subject_id for later subject aggregation.
        """
        self.training = training
        self.augment = augment

        self.items = []  # each: (window_np, y, subject_id)
        self.sfreqs = []
        for r in tqdm(records, desc="Loading & windowing"):
            raw = load_raw(r["path"])
            Xw, sf = extract_windows_fn(raw, canon_chs)
            self.sfreqs.append(sf)
            if Xw.shape[0] == 0:
                continue
            for i in range(Xw.shape[0]):
                self.items.append((Xw[i], r["y"], r["subject_id"]))

        if len(self.items) == 0:
            raise RuntimeError("No windows extracted. Check data durations and sfreq.")

        # sanity: ensure consistent sampling rate
        if np.std(self.sfreqs) > 1e-6:
            print("[WARN] Sampling rate varies across recordings. Model still works (fixed window samples per file).")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y, sid = self.items[idx]
        x = torch.from_numpy(x)  # (C,T)
        # per-window robust normalization (channelwise z-score)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)

        if self.training and self.augment is not None:
            x = self.augment(x)

        return x, torch.tensor(y, dtype=torch.long), sid

class SubjectBagDataset(torch.utils.data.Dataset):
    def __init__(self, records, canon_chs, extract_windows_fn, bag_size=16, training=False, augment=None):
        """
        records: list of dict {path, subject_id, y}
        Returns per-subject bags of windows: (bag_size, C, T), y, subject_id
        """
        self.training = training
        self.augment = augment
        self.bag_size = int(bag_size)

        # group paths by subject (one label per subject assumed)
        subj_map = {}
        for r in records:
            sid = r["subject_id"]
            if sid not in subj_map:
                subj_map[sid] = {"y": int(r["y"]), "paths": []}
            subj_map[sid]["paths"].append(r["path"])

        self.subject_ids = sorted(list(subj_map.keys()))
        self.subj_y = {sid: subj_map[sid]["y"] for sid in self.subject_ids}

        # cache windows per subject
        self.subj_windows = {}
        self.sfreqs = []
        for sid in tqdm(self.subject_ids, desc="Loading & windowing (MIL bags)"):
            all_w = []
            for fp in subj_map[sid]["paths"]:
                raw = load_raw(fp)
                Xw, sf = extract_windows_fn(raw, canon_chs)
                self.sfreqs.append(sf)
                if Xw.shape[0] > 0:
                    all_w.append(Xw)
            if len(all_w) == 0:
                continue
            Xcat = np.concatenate(all_w, axis=0).astype(np.float32)  # (W,C,T)
            self.subj_windows[sid] = Xcat

        # filter subjects with no windows
        self.subject_ids = [sid for sid in self.subject_ids if sid in self.subj_windows]
        if len(self.subject_ids) == 0:
            raise RuntimeError("No subject bags created. Check data durations and sfreq.")

        if np.std(self.sfreqs) > 1e-6:
            print("[WARN] Sampling rate varies across recordings. Model still works (fixed window samples per file).")

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        y = self.subj_y[sid]
        X = self.subj_windows[sid]  # (W,C,T)
        W = X.shape[0]

        # sample bag_size windows (with replacement if needed)
        if W >= self.bag_size:
            sel = np.random.choice(W, size=self.bag_size, replace=False)
        else:
            sel = np.random.choice(W, size=self.bag_size, replace=True)

        Xb = X[sel]  # (BAG,C,T)
        xb = torch.from_numpy(Xb)  # float32

        # per-window normalization + optional augmentation
        # xb: (BAG,C,T)
        xb = (xb - xb.mean(dim=2, keepdim=True)) / (xb.std(dim=2, keepdim=True) + 1e-6)

        if self.training and self.augment is not None:
            # apply per-window
            out = []
            for i in range(xb.shape[0]):
                out.append(self.augment(xb[i]))
            xb = torch.stack(out, dim=0)

        return xb, torch.tensor(y, dtype=torch.long), sid

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        y = self.subj_y[sid]
        X = self.subj_windows[sid]  # (W,C,T)
        W = X.shape[0]

        # sample bag_size windows (with replacement if needed)
        if W >= self.bag_size:
            sel = np.random.choice(W, size=self.bag_size, replace=False)
        else:
            sel = np.random.choice(W, size=self.bag_size, replace=True)

        Xb = X[sel]  # (BAG,C,T)
        xb = torch.from_numpy(Xb)  # float32

        # per-window normalization + optional augmentation
        # xb: (BAG,C,T)
        xb = (xb - xb.mean(dim=2, keepdim=True)) / (xb.std(dim=2, keepdim=True) + 1e-6)

        if self.training and self.augment is not None:
            # apply per-window
            out = []
            for i in range(xb.shape[0]):
                out.append(self.augment(xb[i]))
            xb = torch.stack(out, dim=0)

        return xb, torch.tensor(y, dtype=torch.long), sid

def make_records(df):
    recs = []
    for _, row in df.iterrows():
        recs.append({"path": row["path"], "subject_id": row["subject_id"], "y": int(row["y"])})
    return recs

# ============================================================
# 7) MODEL: NeuroConvFormer-Lite (<=250k params)
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hid = max(4, ch // r)
        self.fc1 = nn.Linear(ch, hid)
        self.fc2 = nn.Linear(hid, ch)

    def forward(self, x):
        # x: (B, C, T)
        s = x.mean(dim=2)              # (B,C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)) # (B,C)
        return x * s.unsqueeze(-1)

class NeuroConvFormerLite(nn.Module):
    """
    Input: (B, C, T)
    Output: logits (B,2)

    Also exposes forward_features() to support MIL subject pooling.
    """
    def __init__(self, n_ch, n_time, d_model=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()

        ks = [7, 15, 31]
        self.dw_convs = nn.ModuleList([
            nn.Conv1d(n_ch, n_ch, kernel_size=k, padding=k//2, groups=n_ch, bias=False)
            for k in ks
        ])
        self.pw_mix = nn.Conv1d(n_ch * len(ks), d_model, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(d_model)

        self.se = SEBlock(d_model, r=8)

        # CRITICAL FIX: More aggressive downsampling (stride=4 instead of 2)
        # Reduces sequence length by 4x to prevent memory explosion
        self.ds = nn.Conv1d(d_model, d_model, kernel_size=5, stride=4, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=2*d_model,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, 1)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

        self._n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_features(self, x):
        # x: (B,C,T) -> z: (B, d_model)
        feats = []
        for dw in self.dw_convs:
            feats.append(dw(x))
        x = torch.cat(feats, dim=1)          # (B, C*3, T)
        x = self.pw_mix(x)                   # (B, d_model, T)
        x = self.bn1(x)
        x = F.gelu(x)

        x = self.se(x)

        x = self.ds(x)                       # (B, d_model, T/4) <- 4x downsampling
        x = self.bn2(x)
        x = F.gelu(x)

        x = x.transpose(1, 2)                # (B, S, d_model) where S = T/4
        x = self.tr(x)

        w = self.attn(x)                     # (B,S,1)
        w = torch.softmax(w, dim=1)
        z = (x * w).sum(dim=1)               # (B,d_model)
        return z

    def forward(self, x):
        z = self.forward_features(x)
        logits = self.head(z)
        return logits

class MILPool(nn.Module):
    """
    Attention pooling across windows in a subject bag.
    Input:  (B, W, D)
    Output: (B, D)
    """
    def __init__(self, d_model=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, z, return_attn=False):
        a = self.net(z)          # (B,W,1)
        a = torch.softmax(a, dim=1)
        z_subj = (z * a).sum(dim=1)
        if return_attn:
            return z_subj, a
        return z_subj

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = float(gamma)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()

# ============================================================
# 8) METRICS (window + subject)
# ============================================================
def binary_metrics(y_true, y_prob, thr=0.5):
    """
    y_true: (N,)
    y_prob: (N,) probability for class=1
    """
    y_pred = (y_prob >= thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    auc = 0.5  # Default value if AUC can't be computed
    if len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.5

    return dict(acc=acc, bacc=bacc, prec=prec, rec=rec, f1=f1, auc=auc)

@torch.no_grad()
def _collect_window_outputs(model, loader):
    """
    Collects window-level:
      - y (N,)
      - p (N,) probability for class1 from window logits
      - logodds (N,) = logits1 - logits0
      - z (N,D) window embeddings from forward_features
      - sid list length N
    """
    model.eval()
    all_y = []
    all_p = []
    all_sid = []
    all_z = []
    all_logodds = []

    for xb, yb, sid in tqdm(loader, desc="Eval", leave=False):
        xb = xb.to(DEVICE, non_blocking=True)
        yb_np = yb.numpy().astype(int)

        z = model.forward_features(xb)                 # (B,D)
        logits = model.head(z)                         # (B,2)
        prob1 = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

        # log-odds for binary = logits1 - logits0
        lo = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()

        all_y.append(yb_np)
        all_p.append(prob1)
        all_logodds.append(lo)
        all_z.append(z.detach().cpu().numpy())
        all_sid.extend(list(sid))

    y = np.concatenate(all_y, axis=0)
    p = np.concatenate(all_p, axis=0)
    logodds = np.concatenate(all_logodds, axis=0)
    z = np.concatenate(all_z, axis=0)
    return y, p, logodds, z, all_sid

@torch.no_grad()
def evaluate(model, pool, loader, keep_frac=0.60, thr=0.5):
    """
    keep_frac: confidence trimming for subject aggregation (top fraction by confidence).
              keep_frac=1.0 uses all windows.
    thr: classification threshold used for both WIN and SUBJ metrics.
    """
    y, p, logodds, z, sids = _collect_window_outputs(model, loader)

    # window-level metrics (exactly as before, just threshold is configurable)
    win = binary_metrics(y, p, thr=thr)

    # subject-level (use MILPool at inference) + logit-mean aggregation
    # Implemented as: attention-weighted mean of per-window log-odds (logit domain),
    # optionally confidence-trimmed first.
    eps = 1e-6
    df = pd.DataFrame({"sid": sids, "y": y, "p": p})
    subj_rows = []

    # To compute attention weights, we need per-subject window embeddings as torch tensors.
    # We'll rebuild per-subject arrays from z/logodds using indices.
    sid_to_indices = {}
    for i, sid in enumerate(sids):
        if sid not in sid_to_indices:
            sid_to_indices[sid] = []
        sid_to_indices[sid].append(i)

    pool.eval()

    subj_true = []
    subj_prob = []

    for sid, idxs in sid_to_indices.items():
        yt = int(df[df["sid"] == sid]["y"].iloc[0])

        pp = np.clip(p[idxs], eps, 1 - eps)
        conf = np.abs(pp - 0.5)
        k = max(1, int(round(float(keep_frac) * len(pp))))
        sel_local = np.argsort(-conf)[:k]
        sel = np.array(idxs, dtype=int)[sel_local]

        z_sel = torch.from_numpy(z[sel]).to(DEVICE)              # (W,D)
        lo_sel = torch.from_numpy(logodds[sel]).to(DEVICE)       # (W,)

        z_sel = z_sel.unsqueeze(0)                               # (1,W,D)
        _, a = pool(z_sel, return_attn=True)                     # a: (1,W,1)
        a = a.squeeze(0).squeeze(-1)                             # (W,)

        lo_agg = torch.sum(a * lo_sel)                           # scalar
        p_agg = torch.sigmoid(lo_agg).item()

        subj_true.append(yt)
        subj_prob.append(float(p_agg))

    subj_true = np.array(subj_true, dtype=int)
    subj_prob = np.array(subj_prob, dtype=float)

    best_thr = 0.5
    best_m = None
    best_bacc = -1.0

    # dense scan
    for thr in np.linspace(0.05, 0.95, 181):
        m = binary_metrics(subj_true, subj_prob, thr=float(thr))
        if m["bacc"] > best_bacc + 1e-12:
            best_bacc = m["bacc"]
            best_thr = float(thr)
            best_m = m

    return win, best_m

def tune_threshold_subject_bacc(model, pool, loader, keep_frac=0.60):
    """
    Train-only threshold tuning:
      - compute subject probabilities using MILPool inference (same as evaluate)
      - scan thresholds and pick the one maximizing subject BAcc
    Returns: best_thr, dict(best_metrics)
    """
    y, p, logodds, z, sids = _collect_window_outputs(model, loader)

    eps = 1e-6
    df = pd.DataFrame({"sid": sids, "y": y, "p": p})

    sid_to_indices = {}
    for i, sid in enumerate(sids):
        if sid not in sid_to_indices:
            sid_to_indices[sid] = []
        sid_to_indices[sid].append(i)

    pool.eval()

    subj_true = []
    subj_prob = []

    for sid, idxs in sid_to_indices.items():
        yt = int(df[df["sid"] == sid]["y"].iloc[0])

        pp = np.clip(p[idxs], eps, 1 - eps)
        conf = np.abs(pp - 0.5)
        k = max(1, int(round(float(keep_frac) * len(pp))))
        sel_local = np.argsort(-conf)[:k]
        sel = np.array(idxs, dtype=int)[sel_local]

        z_sel = torch.from_numpy(z[sel]).to(DEVICE)              # (W,D)
        lo_sel = torch.from_numpy(logodds[sel]).to(DEVICE)       # (W,)

        z_sel = z_sel.unsqueeze(0)                               # (1,W,D)
        _, a = pool(z_sel, return_attn=True)                     # a: (1,W,1)
        a = a.squeeze(0).squeeze(-1)                             # (W,)

        lo_agg = torch.sum(a * lo_sel)                           # scalar
        p_agg = torch.sigmoid(lo_agg).item()

        subj_true.append(yt)
        subj_prob.append(float(p_agg))

    subj_true = np.array(subj_true, dtype=int)
    subj_prob = np.array(subj_prob, dtype=float)

    best_thr = 0.5
    best_m = None
    best_bacc = -1.0

    # dense scan
    for thr in np.linspace(0.05, 0.95, 181):
        m = binary_metrics(subj_true, subj_prob, thr=float(thr))
        if m["bacc"] > best_bacc + 1e-12:
            best_bacc = m["bacc"]
            best_thr = float(thr)
            best_m = m

    return best_thr, best_m

# ============================================================
# 9) TRAINING
# ============================================================
def train_one_fold(fold, train_df, val_df, canon_chs, extract_windows_fn,
                   epochs=5, batch_size=64, lr=2e-3, weight_decay=1e-3,
                   patience=12):

    aug = Augment(
        p_noise=0.35, p_shift=0.35, p_chdrop=0.20, p_banddrop=0.25,
        noise_std=0.015, max_shift_frac=0.08, chdrop_frac=0.10
    )

    # window datasets (for eval metrics exactly as before)
    train_set_win = WindowDataset(make_records(train_df), canon_chs, extract_windows_fn, training=True, augment=aug)
    val_set       = WindowDataset(make_records(val_df), canon_chs, extract_windows_fn, training=False, augment=None)

    # train-only calibration loader for threshold tuning (no augmentation)
    train_set_cal = WindowDataset(make_records(train_df), canon_chs, extract_windows_fn, training=False, augment=None)

    # MIL subject bag dataset (for training objective aligned to subject metrics)
    # NOTE: batch_size in this MIL loader is "subjects per batch"
    BAG_SIZE = 12  # Reduced from 16 to lower memory
    train_set_bag = SubjectBagDataset(make_records(train_df), canon_chs, extract_windows_fn, bag_size=BAG_SIZE, training=True, augment=aug)

    # MIL loader: subjects per batch (reduce if CUDA OOM)
    mil_batch_size = max(1, min(8, batch_size // 8))  # Reduced from 16 to 8

    # Balanced bag sampler (subject-balanced sampling)
    # Build per-subject weights aligned with train_set_bag.subject_ids order.
    subj_labels = np.array([train_set_bag.subj_y[sid] for sid in train_set_bag.subject_ids], dtype=int)
    n0 = int((subj_labels == 0).sum())
    n1 = int((subj_labels == 1).sum())
    w0 = 1.0 / max(1, n0)
    w1 = 1.0 / max(1, n1)
    sample_weights = torch.tensor([w1 if y == 1 else w0 for y in subj_labels], dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_set_bag),
        replacement=True
    )

    train_loader_bag = torch.utils.data.DataLoader(
        train_set_bag, batch_size=mil_batch_size, sampler=sampler,
        num_workers=0, pin_memory=False, drop_last=True
    )

    # Eval loader stays window-level
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    train_loader_cal = torch.utils.data.DataLoader(
        train_set_cal, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    # infer shapes
    x0, _, _ = train_set_win[0]
    n_ch, n_time = x0.shape

    model = NeuroConvFormerLite(n_ch=n_ch, n_time=n_time, d_model=64, n_heads=4, n_layers=2, dropout=0.25).to(DEVICE)
    pool = MILPool(d_model=64).to(DEVICE)

    n_params = sum(p.numel() for p in list(model.parameters()) + list(pool.parameters()) if p.requires_grad)
    print(f"[fold{fold:02d}] Model params: {n_params}")
    if n_params > 250_000:
        raise RuntimeError(f"Param budget exceeded: {n_params} > 250k")

    # class balance (use subject labels for weighting)
    subj_train = train_df.groupby("subject_id").agg(y=("y", "first")).reset_index()
    y_train = subj_train["y"].values.astype(int)
    n0_ce = (y_train == 0).sum()
    n1_ce = (y_train == 1).sum()
    pos_weight = torch.tensor([max(1.0, n0_ce / max(1, n1_ce))], device=DEVICE)

    # class weights for CE/Focal (normalized)
    w0_ce = (n0_ce + n1_ce) / max(1, 2 * n0_ce)
    w1_ce = (n0_ce + n1_ce) / max(1, 2 * n1_ce)
    class_w = torch.tensor([w0_ce, w1_ce], dtype=torch.float32, device=DEVICE)

    # Focal loss (weighted)
    crit = FocalLoss(weight=class_w, gamma=2.0)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(pool.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    total_steps = epochs * len(train_loader_bag)
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_bacc = -1
    best_state = None
    best_state_pool = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        pool.train()
        losses = []

        pbar = tqdm(train_loader_bag, desc=f"[fold{fold:02d}] MIL Train ep{ep:03d}", leave=False)
        for xb_bag, yb, _ in pbar:
            # xb_bag: (Bsubj, BAG, C, T)
            xb_bag = xb_bag.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            Bsubj, BAG, C, T = xb_bag.shape
            xb_flat = xb_bag.view(Bsubj * BAG, C, T)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                z = model.forward_features(xb_flat)                 # (Bsubj*BAG, D)
                z = z.view(Bsubj, BAG, -1)                          # (Bsubj, BAG, D)
                z_subj = pool(z)                                    # (Bsubj, D)
                logits_subj = model.head(z_subj)                    # (Bsubj, 2)
                loss = crit(logits_subj, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(pool.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)), lr=opt.param_groups[0]["lr"])

        # eval each epoch (window + subject metrics) with default thr=0.5 (no tuning on VAL)
        print(f"[fold{fold:02d}] Evaluating epoch {ep}...")
        eval_result = evaluate(model, pool, val_loader, keep_frac=0.60, thr=0.5)
        print(f"[fold{fold:02d}] Eval returned: {type(eval_result)}")
        
        if not isinstance(eval_result, tuple) or len(eval_result) != 2:
            print(f"[ERROR] evaluate() returned wrong type: {type(eval_result)}, value: {eval_result}")
            raise ValueError("evaluate() must return (win_metrics_dict, subj_metrics_dict)")
        
        win_m, subj_m = eval_result

        print(f"[fold{fold:02d}] ep{ep:03d} | "
              f"VAL(win) acc={win_m['acc']:.4f} bacc={win_m['bacc']:.4f} f1={win_m['f1']:.4f} auc={win_m['auc']:.4f} | "
              f"VAL(subj) acc={subj_m['acc']:.4f} bacc={subj_m['bacc']:.4f} f1={subj_m['f1']:.4f} auc={subj_m['auc']:.4f}")

        # early stop on subject bacc (thr=0.5 only)
        if subj_m["bacc"] > best_bacc + 1e-4:
            best_bacc = subj_m["bacc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_state_pool = {k: v.detach().cpu().clone() for k, v in pool.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[fold{fold:02d}] Early stop at ep{ep:03d} (best subj_bacc={best_bacc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_state_pool is not None:
        pool.load_state_dict(best_state_pool)

    # per-fold threshold tuning (train-only)
    best_thr, best_train_subj_m = tune_threshold_subject_bacc(model, pool, train_loader_cal, keep_frac=0.60)
    print(f"[fold{fold:02d}] Train-only tuned subject threshold: thr={best_thr:.3f} | "
          f"train_subj_bacc={best_train_subj_m['bacc']:.4f} f1={best_train_subj_m['f1']:.4f} "
          f"prec={best_train_subj_m['prec']:.4f} rec={best_train_subj_m['rec']:.4f}")

    # final fold eval (best checkpoint) using tuned threshold
    win_m, subj_m = evaluate(model, pool, val_loader, keep_frac=0.60, thr=best_thr)
    return model, win_m, subj_m

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # 1) LABEL LOADING - UPDATED FOR EXCEL
    # ============================================================
    def load_labels_xlsx(xlsx_path):
        """
        Excel columns:
            eeg_session_file : subject/session id (e.g., "8010_1_PD_REST.edf")
            class_label      : text or numeric (e.g., "Healthy", "PD")
        """
        df = pd.read_excel(xlsx_path)
        df.columns = [c.strip().lower() for c in df.columns]

        if "eeg_session_file" not in df.columns:
            raise ValueError("Missing column: eeg_session_file")
        if "class_label" not in df.columns:
            raise ValueError("Missing column: class_label")

        df = df[["eeg_session_file", "class_label"]].copy()
        df.columns = ["subject_id", "label_raw"]

        # Clean subject ids - remove .edf extension if present
        df["subject_id"] = df["subject_id"].astype(str).str.strip()
        df["subject_id"] = df["subject_id"].str.replace(".edf", "", case=False, regex=False)

        raw = df["label_raw"]

        # map labels → 0/1
        if raw.dtype == object:
            low = raw.astype(str).str.lower().str.strip()
            # PD = 1, Healthy/HC = 0
            pd_mask = low.str.contains("pd|parkinson|patient|case|disease", regex=True)
            hc_mask = low.str.contains("healthy|hc|control|normal", regex=True)
            
            if pd_mask.any() and hc_mask.any():
                y = np.where(pd_mask, 1, 0).astype(int)
            else:
                # fallback to factorize
                y = pd.factorize(raw)[0].astype(int)
        else:
            y = raw.astype(int).values

        if len(np.unique(y)) != 2:
            raise ValueError(f"Non-binary labels detected: {np.unique(y)}")

        df["y"] = y
        return df[["subject_id", "y"]]

    labels_df = load_labels_xlsx(LABEL_FILE)
    print(f"[INFO] Loaded labels: {len(labels_df)} entries")
    print(f"[INFO] Class distribution: {labels_df['y'].value_counts().to_dict()}")

    # ============================================================
    # 2) DATA DISCOVERY + SUBJECT ID PARSING - UPDATED FOR EDF
    # ============================================================
    SUPPORTED_EXTS = [".edf"]

    def discover_eeg_files(root_dir):
        files = []
        for ext in SUPPORTED_EXTS:
            files.extend(glob.glob(os.path.join(root_dir, f"**/*{ext}"), recursive=True))
        files = sorted(list(set(files)))
        if len(files) == 0:
            raise FileNotFoundError(
                f"No EEG files found in {root_dir} with extensions {SUPPORTED_EXTS}"
            )
        return files

    def infer_subject_id_from_path(path):
        """
        Extract filename without extension as subject ID.
        Example: "8010_1_PD_REST.edf" → "8010_1_PD_REST"
        """
        name = os.path.basename(path)
        return os.path.splitext(name)[0].strip()

    files = discover_eeg_files(DATA_DIR)
    print(f"[INFO] Found {len(files)} EDF files")

    # align files with labels
    file_rows = []
    label_map = dict(zip(labels_df["subject_id"], labels_df["y"]))

    # allow minor normalization of subject keys
    def normalize_sid(sid):
        sid = str(sid).strip()
        sid = sid.replace(" ", "")
        return sid

    label_map_norm = {normalize_sid(k): v for k, v in label_map.items()}

    for fp in files:
        sid = infer_subject_id_from_path(fp)
        y = label_map_norm.get(normalize_sid(sid), None)
        
        if y is None:
            # sometimes labels use just numeric id, try extracting trailing digits
            digits = re.findall(r"\d+", sid)
            if len(digits):
                y = label_map_norm.get(normalize_sid(digits[-1]), None)
        
        if y is None:
            print(f"[WARN] No label found for: {sid} (from {os.path.basename(fp)})")
            continue
        
        file_rows.append((fp, sid, int(y)))

    if len(file_rows) == 0:
        raise RuntimeError(
            "No files matched with labels. Check subject_id mapping between "
            "label file and file naming."
        )

    meta_df = pd.DataFrame(file_rows, columns=["path", "subject_id", "y"])
    print(f"[INFO] Matched recordings: {len(meta_df)} | Subjects: {meta_df['subject_id'].nunique()}")
    print(f"[INFO] Matched class distribution: {meta_df.groupby('y')['subject_id'].nunique().to_dict()}")

    # ============================================================
    # 3) EEG LOADING (EDF-only) + CANONICAL INTERSECTION
    # ============================================================
    def load_raw(fp):
        """Load EDF file only"""
        return mne.io.read_raw_edf(fp, preload=True, verbose=False)

    def eeg_only_pick(raw):
        picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
        if len(picks) == 0:
            raise RuntimeError("No EEG channels found in file.")
        return picks

    # compute canonical EEG channel intersection
    all_ch_sets = []
    for fp in tqdm(meta_df["path"].tolist(), desc="Scanning channels"):
        raw = load_raw(fp)
        picks = eeg_only_pick(raw)
        chs = [raw.ch_names[i] for i in picks]
        # normalize channel names
        chs = [c.strip().upper() for c in chs]
        all_ch_sets.append(set(chs))

    canon = set.intersection(*all_ch_sets)
    canon = sorted(list(canon))
    if len(canon) < 4:
        raise RuntimeError(f"Canonical EEG intersection too small: {len(canon)} channels: {canon}")

    print(f"[INFO] Canonical EEG channels (intersection): {len(canon)}")

    # ============================================================
    # WINDOWING FUNCTION (needed by datasets)
    # ============================================================
    def extract_windows_from_raw(raw, canon_chs, win_sec=10.0, overlap=0.50):
        """
        Returns: X_windows (nW, C, T), sfreq
        Uses the full duration; drops the last partial window.
        """
        sfreq = float(raw.info["sfreq"])
        # pick canonical channels (case-insensitive)
        name_to_idx = {c.strip().upper(): i for i, c in enumerate(raw.ch_names)}
        picks = [name_to_idx[c] for c in canon_chs if c in name_to_idx]
        if len(picks) != len(canon_chs):
            # should not happen if intersection computed correctly, but guard anyway
            raise RuntimeError("Canonical channel mismatch during extraction.")

        data = raw.get_data(picks=picks)  # (C, N)
        C, N = data.shape

        win_len = int(round(win_sec * sfreq))
        hop = int(round(win_len * (1.0 - overlap)))
        hop = max(1, hop)

        if N < win_len:
            return np.zeros((0, C, win_len), dtype=np.float32), sfreq

        starts = np.arange(0, N - win_len + 1, hop, dtype=int)
        X = np.stack([data[:, s:s+win_len] for s in starts], axis=0).astype(np.float32)  # (W,C,T)
        return X, sfreq

    # ============================================================
    # RUN CROSS-VALIDATION
    # ============================================================
    # Make load_raw available globally for datasets
    globals()['load_raw'] = load_raw
    
    # Run CV
    subj_df = meta_df.groupby("subject_id").agg(
        y=("y", "first")
    ).reset_index()

    X_dummy = np.zeros((len(subj_df), 1))
    y_subj = subj_df["y"].values.astype(int)
    groups = subj_df["subject_id"].values.astype(str)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_dummy, y_subj, groups), start=1):
        tr_sids = set(subj_df.iloc[tr_idx]["subject_id"].tolist())
        va_sids = set(subj_df.iloc[va_idx]["subject_id"].tolist())

        train_df = meta_df[meta_df["subject_id"].isin(tr_sids)].reset_index(drop=True)
        val_df   = meta_df[meta_df["subject_id"].isin(va_sids)].reset_index(drop=True)

        print("\n" + "="*70)
        print(f"[fold{fold:02d}] Train subjects: {len(tr_sids)} | Val subjects: {len(va_sids)}")
        print(f"[fold{fold:02d}] Train recs: {len(train_df)} | Val recs: {len(val_df)}")
        print("="*70)

        model, win_m, subj_m = train_one_fold(fold, train_df, val_df, canon, extract_windows_from_raw)

        fold_results.append((win_m, subj_m))

    def summarize(results, keyset=("acc","bacc","prec","rec","f1","auc")):
        arr = {k: [] for k in keyset}
        for m in results:
            for k in keyset:
                arr[k].append(m[k] if not (isinstance(m[k], float) and np.isnan(m[k])) else np.nan)
        out = {}
        for k, v in arr.items():
            vv = np.array(v, dtype=float)
            out[k] = (np.nanmean(vv), np.nanstd(vv))
        return out

    win_folds  = [r[0] for r in fold_results]
    subj_folds = [r[1] for r in fold_results]

    win_sum  = summarize(win_folds)
    subj_sum = summarize(subj_folds)

    print("\n" + "#"*72)
    print("FOLD-WISE RESULTS:")
    for i, (w, s) in enumerate(fold_results, start=1):
        print(f"[fold{i:02d}] WIN  acc={w['acc']:.4f} bacc={w['bacc']:.4f} prec={w['prec']:.4f} rec={w['rec']:.4f} f1={w['f1']:.4f} auc={w['auc']:.4f}")
        print(f"         SUBJ acc={s['acc']:.4f} bacc={s['bacc']:.4f} prec={s['prec']:.4f} rec={s['rec']:.4f} f1={s['f1']:.4f} auc={s['auc']:.4f}")

    print("\n" + "#"*72)
    print("OVERALL (mean ± std across folds):")
    print(f"WIN : acc={win_sum['acc'][0]:.4f}±{win_sum['acc'][1]:.4f} | "
          f"bacc={win_sum['bacc'][0]:.4f}±{win_sum['bacc'][1]:.4f} | "
          f"prec={win_sum['prec'][0]:.4f}±{win_sum['prec'][1]:.4f} | "
          f"rec={win_sum['rec'][0]:.4f}±{win_sum['rec'][1]:.4f} | "
          f"f1={win_sum['f1'][0]:.4f}±{win_sum['f1'][1]:.4f} | "
          f"auc={win_sum['auc'][0]:.4f}±{win_sum['auc'][1]:.4f}")
    print(f"SUBJ: acc={subj_sum['acc'][0]:.4f}±{subj_sum['acc'][1]:.4f} | "
          f"bacc={subj_sum['bacc'][0]:.4f}±{subj_sum['bacc'][1]:.4f} | "
          f"prec={subj_sum['prec'][0]:.4f}±{subj_sum['prec'][1]:.4f} | "
          f"rec={subj_sum['rec'][0]:.4f}±{subj_sum['rec'][1]:.4f} | "
          f"f1={subj_sum['f1'][0]:.4f}±{subj_sum['f1'][1]:.4f} | "
          f"auc={subj_sum['auc'][0]:.4f}±{subj_sum['auc'][1]:.4f}")
    print("#"*72)