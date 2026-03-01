#!/usr/bin/env python3
# ============================================================
# DUAL-BRANCH ARCHITECTURE - Forces Learning of BOTH Classes
# Key Innovation: Separate expert networks for each class
# ============================================================

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import re, glob, math, json, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, roc_auc_score,
    confusion_matrix
)

import mne

# -----------------------
# USER PATHS
# -----------------------
DATA_DIR = r"C:\projects\NCF-lite\dataset\PD REST\edf"
LABEL_FILE = r"C:\projects\NCF-lite\test-2\eeg_sessions_labels.xlsx"
MODEL_SAVE_DIR = r"C:\projects\NCF-lite\models"

# -----------------------
# GLOBALS
# -----------------------
SEED = 42
DEVICE = "cpu"

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(SEED)

WIN_SEC = 10.0
OVERLAP = 0.50

# ============================================================
# AUGMENTATIONS - MINIMAL
# ============================================================
class Augment:
    def __init__(self, p_noise=0.15, p_shift=0.15, noise_std=0.01, max_shift_frac=0.03):
        self.p_noise = p_noise
        self.p_shift = p_shift
        self.noise_std = noise_std
        self.max_shift_frac = max_shift_frac

    def __call__(self, x):
        C, T = x.shape

        if random.random() < self.p_shift:
            max_shift = int(self.max_shift_frac * T)
            if max_shift > 0:
                s = random.randint(-max_shift, max_shift)
                x = torch.roll(x, shifts=s, dims=1)

        if random.random() < self.p_noise:
            x = x + self.noise_std * torch.randn_like(x)

        return x

# ============================================================
# DATASETS
# ============================================================
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, records, canon_chs, extract_windows_fn, training=False, augment=None):
        self.training = training
        self.augment = augment

        self.items = []
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
            raise RuntimeError("No windows extracted.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y, sid = self.items[idx]
        x = torch.from_numpy(x)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)

        if self.training and self.augment is not None:
            x = self.augment(x)

        return x, torch.tensor(y, dtype=torch.long), sid

class SubjectBagDataset(torch.utils.data.Dataset):
    def __init__(self, records, canon_chs, extract_windows_fn, bag_size=8, training=False, augment=None):
        self.training = training
        self.augment = augment
        self.bag_size = int(bag_size)

        subj_map = {}
        for r in records:
            sid = r["subject_id"]
            if sid not in subj_map:
                subj_map[sid] = {"y": int(r["y"]), "paths": []}
            subj_map[sid]["paths"].append(r["path"])

        self.subject_ids = sorted(list(subj_map.keys()))
        self.subj_y = {sid: subj_map[sid]["y"] for sid in self.subject_ids}

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
            Xcat = np.concatenate(all_w, axis=0).astype(np.float32)
            self.subj_windows[sid] = Xcat

        self.subject_ids = [sid for sid in self.subject_ids if sid in self.subj_windows]
        if len(self.subject_ids) == 0:
            raise RuntimeError("No subject bags created.")

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        y = self.subj_y[sid]
        X = self.subj_windows[sid]
        W = X.shape[0]

        if W >= self.bag_size:
            sel = np.random.choice(W, size=self.bag_size, replace=False)
        else:
            sel = np.random.choice(W, size=self.bag_size, replace=True)

        Xb = X[sel]
        xb = torch.from_numpy(Xb)
        xb = (xb - xb.mean(dim=2, keepdim=True)) / (xb.std(dim=2, keepdim=True) + 1e-6)

        if self.training and self.augment is not None:
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
# DUAL-BRANCH MODEL - CLASS-SPECIFIC EXPERTS
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hid = max(4, ch // r)
        self.fc1 = nn.Linear(ch, hid)
        self.fc2 = nn.Linear(hid, ch)

    def forward(self, x):
        s = x.mean(dim=2)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)

class ClassSpecificExpert(nn.Module):
    """Expert network specialized for one class"""
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)  # Binary score for this class
        )
    
    def forward(self, x):
        return self.net(x)

class DualBranchNeuroConvFormer(nn.Module):
    """
    Dual-branch architecture with class-specific experts.
    Forces model to learn discriminative features for BOTH classes.
    """
    def __init__(self, n_ch, n_time, d_model=56, n_heads=4, n_layers=2, dropout=0.25):
        super().__init__()

        # Shared feature extraction (same as before)
        ks = [7, 15, 31]
        self.dw_convs = nn.ModuleList([
            nn.Conv1d(n_ch, n_ch, kernel_size=k, padding=k//2, groups=n_ch, bias=False)
            for k in ks
        ])
        self.pw_mix = nn.Conv1d(n_ch * len(ks), d_model, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(d_model)

        self.se = SEBlock(d_model, r=8)

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
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )

        # NEW: Class-specific expert branches
        self.class0_expert = ClassSpecificExpert(d_model, dropout=dropout)
        self.class1_expert = ClassSpecificExpert(d_model, dropout=dropout)
        
        # Shared feature refinement
        self.shared_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Final combination layer
        self.final = nn.Linear(64 + 2, 2)  # 64 shared + 2 expert scores

        self._n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_features(self, x):
        """Extract shared features (same as before)"""
        feats = []
        for dw in self.dw_convs:
            feats.append(dw(x))
        x = torch.cat(feats, dim=1)
        x = self.pw_mix(x)
        x = self.bn1(x)
        x = F.gelu(x)

        x = self.se(x)

        x = self.ds(x)
        x = self.bn2(x)
        x = F.gelu(x)

        x = x.transpose(1, 2)
        x = self.tr(x)

        w = self.attn(x)
        w = torch.softmax(w, dim=1)
        z = (x * w).sum(dim=1)
        return z

    def forward(self, x):
        """
        Forward pass with dual-branch experts.
        Each expert scores how likely the input belongs to its class.
        """
        # Get shared features
        z = self.forward_features(x)  # (B, d_model)
        
        # Get class-specific scores from experts
        score_class0 = self.class0_expert(z)  # (B, 1)
        score_class1 = self.class1_expert(z)  # (B, 1)
        
        # Shared feature refinement
        shared = self.shared_head(z)  # (B, 64)
        
        # Combine shared features + expert scores
        combined = torch.cat([shared, score_class0, score_class1], dim=1)  # (B, 66)
        
        # Final classification
        logits = self.final(combined)  # (B, 2)
        
        return logits

class MILPool(nn.Module):
    def __init__(self, d_model=56):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, z, return_attn=False):
        a = self.net(z)
        a = torch.softmax(a, dim=1)
        z_subj = (z * a).sum(dim=1)
        if return_attn:
            return z_subj, a
        return z_subj

# ============================================================
# AUXILIARY LOSS - Encourages expert specialization
# ============================================================
class DualBranchLoss(nn.Module):
    """
    Combined loss:
    1. Main CE loss on final predictions
    2. Expert specialization loss (class0_expert should score high on class0, etc.)
    """
    def __init__(self, weight=None, expert_weight=0.3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.expert_weight = expert_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, target, score_class0, score_class1):
        # Main classification loss
        ce_loss = self.ce(logits, target)
        
        # Expert specialization loss
        # class0_expert should output high scores for class0 samples
        # class1_expert should output high scores for class1 samples
        target_float = target.float().unsqueeze(1)  # (B, 1)
        
        # For class0 samples (target=0), score_class0 should be high (close to 1)
        # For class1 samples (target=1), score_class0 should be low (close to 0)
        expert0_target = 1.0 - target_float
        expert1_target = target_float
        
        expert0_loss = self.bce(score_class0, expert0_target)
        expert1_loss = self.bce(score_class1, expert1_target)
        
        expert_loss = (expert0_loss + expert1_loss) / 2.0
        
        # Combined loss
        total_loss = ce_loss + self.expert_weight * expert_loss
        
        return total_loss

# ============================================================
# METRICS
# ============================================================
def binary_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    auc = 0.5
    if len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return dict(acc=acc, bacc=bacc, prec=prec, rec=rec, f1=f1, auc=auc,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

@torch.no_grad()
def _collect_window_outputs(model, loader):
    model.eval()
    all_y = []
    all_p = []
    all_sid = []
    all_z = []
    all_logodds = []

    for xb, yb, sid in tqdm(loader, desc="Eval", leave=False):
        xb = xb.to(DEVICE)
        yb_np = yb.numpy().astype(int)

        z = model.forward_features(xb)
        logits = model(xb)
        prob1 = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

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
    y, p, logodds, z, sids = _collect_window_outputs(model, loader)

    win = binary_metrics(y, p, thr=thr)

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

        z_sel = torch.from_numpy(z[sel]).to(DEVICE)
        lo_sel = torch.from_numpy(logodds[sel]).to(DEVICE)

        z_sel = z_sel.unsqueeze(0)
        _, a = pool(z_sel, return_attn=True)
        a = a.squeeze(0).squeeze(-1)

        lo_agg = torch.sum(a * lo_sel)
        p_agg = torch.sigmoid(lo_agg).item()

        subj_true.append(yt)
        subj_prob.append(float(p_agg))

    subj_true = np.array(subj_true, dtype=int)
    subj_prob = np.array(subj_prob, dtype=float)

    best_thr = 0.5
    best_m = None
    best_bacc = -1.0

    for thr in np.linspace(0.05, 0.95, 181):
        m = binary_metrics(subj_true, subj_prob, thr=float(thr))
        if m["bacc"] > best_bacc + 1e-12:
            best_bacc = m["bacc"]
            best_thr = float(thr)
            best_m = m

    return win, best_m

def tune_threshold_subject_bacc(model, pool, loader, keep_frac=0.60):
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

        z_sel = torch.from_numpy(z[sel]).to(DEVICE)
        lo_sel = torch.from_numpy(logodds[sel]).to(DEVICE)

        z_sel = z_sel.unsqueeze(0)
        _, a = pool(z_sel, return_attn=True)
        a = a.squeeze(0).squeeze(-1)

        lo_agg = torch.sum(a * lo_sel)
        p_agg = torch.sigmoid(lo_agg).item()

        subj_true.append(yt)
        subj_prob.append(float(p_agg))

    subj_true = np.array(subj_true, dtype=int)
    subj_prob = np.array(subj_prob, dtype=float)

    best_thr = 0.5
    best_m = None
    best_bacc = -1.0

    for thr in np.linspace(0.05, 0.95, 181):
        m = binary_metrics(subj_true, subj_prob, thr=float(thr))
        if m["bacc"] > best_bacc + 1e-12:
            best_bacc = m["bacc"]
            best_thr = float(thr)
            best_m = m

    return best_thr, best_m

# ============================================================
# TRAINING - WITH DUAL-BRANCH LOSS
# ============================================================
def train_one_fold(fold, train_df, val_df, canon_chs, extract_windows_fn,
                   epochs=15, batch_size=8, lr=5e-4, weight_decay=1e-4,
                   patience=10, save_dir=None):

    aug = Augment(p_noise=0.15, p_shift=0.15, noise_std=0.01, max_shift_frac=0.03)

    train_set_win = WindowDataset(make_records(train_df), canon_chs, extract_windows_fn, training=True, augment=aug)
    val_set = WindowDataset(make_records(val_df), canon_chs, extract_windows_fn, training=False, augment=None)
    train_set_cal = WindowDataset(make_records(train_df), canon_chs, extract_windows_fn, training=False, augment=None)

    BAG_SIZE = 8
    train_set_bag = SubjectBagDataset(make_records(train_df), canon_chs, extract_windows_fn, bag_size=BAG_SIZE, training=True, augment=aug)

    mil_batch_size = 4

    subj_labels = np.array([train_set_bag.subj_y[sid] for sid in train_set_bag.subject_ids], dtype=int)
    n0 = int((subj_labels == 0).sum())
    n1 = int((subj_labels == 1).sum())
    
    print(f"[fold{fold:02d}] Training class distribution: Class0={n0}, Class1={n1}")
    
    # BALANCED weighting (not too aggressive)
    ratio = n1 / max(1, n0)  # ~2.0 for your data
    w0 = 1.3  # Moderate boost for minority class in sampling
    w1 = 1.0
    
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

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    train_loader_cal = torch.utils.data.DataLoader(
        train_set_cal, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    x0, _, _ = train_set_win[0]
    n_ch, n_time = x0.shape

    # NEW: Dual-branch model
    model = DualBranchNeuroConvFormer(n_ch=n_ch, n_time=n_time, d_model=56, n_heads=4, n_layers=2, dropout=0.25).to(DEVICE)
    pool = MILPool(d_model=56).to(DEVICE)

    n_params = sum(p.numel() for p in list(model.parameters()) + list(pool.parameters()) if p.requires_grad)
    print(f"[fold{fold:02d}] Model params: {n_params}")

    # BALANCED class weights in loss (not too strong)
    w0_ce = 1.8  # Moderate boost for minority class
    w1_ce = 1.0
    class_w = torch.tensor([w0_ce, w1_ce], dtype=torch.float32, device=DEVICE)
    
    print(f"[fold{fold:02d}] Loss class weights: Class0={w0_ce:.2f}, Class1={w1_ce:.2f}")

    # Dual-branch loss with REDUCED expert weight for more balanced learning
    crit = DualBranchLoss(weight=class_w, expert_weight=0.2)  # Reduced from 0.3

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(pool.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    total_steps = epochs * len(train_loader_bag)
    warmup_steps = max(1, int(0.15 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_bacc = -1
    best_state = None
    best_state_pool = None
    bad = 0
    best_epoch = 0

    for ep in range(1, epochs + 1):
        model.train()
        pool.train()
        losses = []

        pbar = tqdm(train_loader_bag, desc=f"[fold{fold:02d}] Dual-Branch Train ep{ep:03d}", leave=False)
        for xb_bag, yb, _ in pbar:
            xb_bag = xb_bag.to(DEVICE)
            yb = yb.to(DEVICE)

            Bsubj, BAG, C, T = xb_bag.shape
            xb_flat = xb_bag.view(Bsubj * BAG, C, T)

            opt.zero_grad(set_to_none=True)

            # Get shared features
            z = model.forward_features(xb_flat)  # (Bsubj*BAG, d_model)
            z = z.view(Bsubj, BAG, -1)  # (Bsubj, BAG, d_model)
            z_subj = pool(z)  # (Bsubj, d_model)
            
            # Get expert scores for loss
            score_class0 = model.class0_expert(z_subj)  # (Bsubj, 1)
            score_class1 = model.class1_expert(z_subj)  # (Bsubj, 1)
            
            # Get final logits
            shared = model.shared_head(z_subj)
            combined = torch.cat([shared, score_class0, score_class1], dim=1)
            logits_subj = model.final(combined)
            
            # Dual-branch loss (CE + expert specialization)
            loss = crit(logits_subj, yb, score_class0, score_class1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(pool.parameters()), 1.0)
            opt.step()
            sch.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)), lr=opt.param_groups[0]["lr"])

        print(f"[fold{fold:02d}] Evaluating epoch {ep}...")
        win_m, subj_m = evaluate(model, pool, val_loader, keep_frac=0.60, thr=0.5)

        print(f"[fold{fold:02d}] ep{ep:03d} | "
              f"VAL(win) acc={win_m['acc']:.4f} bacc={win_m['bacc']:.4f} f1={win_m['f1']:.4f} auc={win_m['auc']:.4f} "
              f"CM=[TN:{win_m['tn']}, FP:{win_m['fp']}, FN:{win_m['fn']}, TP:{win_m['tp']}]")
        print(f"          VAL(subj) acc={subj_m['acc']:.4f} bacc={subj_m['bacc']:.4f} f1={subj_m['f1']:.4f} auc={subj_m['auc']:.4f} "
              f"CM=[TN:{subj_m['tn']}, FP:{subj_m['fp']}, FN:{subj_m['fn']}, TP:{subj_m['tp']}]")

        if subj_m["bacc"] > best_bacc + 1e-4:
            best_bacc = subj_m["bacc"]
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_state_pool = {k: v.detach().cpu().clone() for k, v in pool.state_dict().items()}
            
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                
                model_name = (f"fold{fold:02d}_epoch{ep:03d}_"
                            f"subjBAcc{subj_m['bacc']:.4f}_"
                            f"subjF1{subj_m['f1']:.4f}_"
                            f"winBAcc{win_m['bacc']:.4f}_DualBranch.pth")
                
                save_path = os.path.join(save_dir, model_name)
                
                checkpoint = {
                    'fold': fold,
                    'epoch': ep,
                    'model_state_dict': best_state,
                    'pool_state_dict': best_state_pool,
                    'optimizer_state_dict': opt.state_dict(),
                    'window_metrics': win_m,
                    'subject_metrics': subj_m,
                    'n_params': n_params,
                    'model_config': {
                        'd_model': 56,
                        'n_heads': 4,
                        'n_layers': 2,
                        'n_ch': n_ch,
                        'n_time': n_time,
                        'architecture': 'DualBranch'
                    },
                    'canonical_channels': canon_chs
                }
                
                torch.save(checkpoint, save_path)
                print(f"[fold{fold:02d}] ✓ Model saved: {model_name}")
            
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[fold{fold:02d}] Early stop at ep{ep:03d} (best subj_bacc={best_bacc:.4f} at epoch {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_state_pool is not None:
        pool.load_state_dict(best_state_pool)

    best_thr, best_train_subj_m = tune_threshold_subject_bacc(model, pool, train_loader_cal, keep_frac=0.60)
    print(f"[fold{fold:02d}] Train-only tuned subject threshold: thr={best_thr:.3f} | "
          f"train_subj_bacc={best_train_subj_m['bacc']:.4f} f1={best_train_subj_m['f1']:.4f} "
          f"prec={best_train_subj_m['prec']:.4f} rec={best_train_subj_m['rec']:.4f}")

    win_m, subj_m = evaluate(model, pool, val_loader, keep_frac=0.60, thr=best_thr)
    return model, win_m, subj_m

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print(f"[INFO] Running DUAL-BRANCH architecture on CPU with {torch.get_num_threads()} threads")
    print("[INFO] Architecture: Class-specific expert networks")
    
    def load_labels_xlsx(xlsx_path):
        df = pd.read_excel(xlsx_path)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df[["eeg_session_file", "class_label"]].copy()
        df.columns = ["subject_id", "label_raw"]
        df["subject_id"] = df["subject_id"].astype(str).str.strip()
        df["subject_id"] = df["subject_id"].str.replace(".edf", "", case=False, regex=False)
        raw = df["label_raw"]
        if raw.dtype == object:
            low = raw.astype(str).str.lower().str.strip()
            pd_mask = low.str.contains("pd|parkinson|patient|case|disease", regex=True)
            hc_mask = low.str.contains("healthy|hc|control|normal", regex=True)
            if pd_mask.any() and hc_mask.any():
                y = np.where(pd_mask, 1, 0).astype(int)
            else:
                y = pd.factorize(raw)[0].astype(int)
        else:
            y = raw.astype(int).values
        df["y"] = y
        return df[["subject_id", "y"]]

    labels_df = load_labels_xlsx(LABEL_FILE)
    print(f"[INFO] Loaded labels: {len(labels_df)} entries")
    print(f"[INFO] Class distribution: {labels_df['y'].value_counts().to_dict()}")

    SUPPORTED_EXTS = [".edf"]

    def discover_eeg_files(root_dir):
        files = []
        for ext in SUPPORTED_EXTS:
            files.extend(glob.glob(os.path.join(root_dir, f"**/*{ext}"), recursive=True))
        return sorted(list(set(files)))

    def infer_subject_id_from_path(path):
        return os.path.splitext(os.path.basename(path))[0].strip()

    files = discover_eeg_files(DATA_DIR)
    print(f"[INFO] Found {len(files)} EDF files")

    label_map = dict(zip(labels_df["subject_id"], labels_df["y"]))
    def normalize_sid(sid):
        return str(sid).strip().replace(" ", "")
    label_map_norm = {normalize_sid(k): v for k, v in label_map.items()}

    file_rows = []
    for fp in files:
        sid = infer_subject_id_from_path(fp)
        y = label_map_norm.get(normalize_sid(sid), None)
        if y is None:
            digits = re.findall(r"\d+", sid)
            if len(digits):
                y = label_map_norm.get(normalize_sid(digits[-1]), None)
        if y is not None:
            file_rows.append((fp, sid, int(y)))

    meta_df = pd.DataFrame(file_rows, columns=["path", "subject_id", "y"])
    print(f"[INFO] Matched recordings: {len(meta_df)} | Subjects: {meta_df['subject_id'].nunique()}")

    def load_raw(fp):
        return mne.io.read_raw_edf(fp, preload=True, verbose=False)

    def eeg_only_pick(raw):
        picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
        return picks

    all_ch_sets = []
    for fp in tqdm(meta_df["path"].tolist(), desc="Scanning channels"):
        raw = load_raw(fp)
        picks = eeg_only_pick(raw)
        chs = [raw.ch_names[i].strip().upper() for i in picks]
        all_ch_sets.append(set(chs))

    canon = sorted(list(set.intersection(*all_ch_sets)))
    print(f"[INFO] Canonical EEG channels: {len(canon)}")

    def extract_windows_from_raw(raw, canon_chs, win_sec=10.0, overlap=0.50):
        sfreq = float(raw.info["sfreq"])
        name_to_idx = {c.strip().upper(): i for i, c in enumerate(raw.ch_names)}
        picks = [name_to_idx[c] for c in canon_chs if c in name_to_idx]
        data = raw.get_data(picks=picks)
        C, N = data.shape
        win_len = int(round(win_sec * sfreq))
        hop = int(round(win_len * (1.0 - overlap)))
        hop = max(1, hop)
        if N < win_len:
            return np.zeros((0, C, win_len), dtype=np.float32), sfreq
        starts = np.arange(0, N - win_len + 1, hop, dtype=int)
        X = np.stack([data[:, s:s+win_len] for s in starts], axis=0).astype(np.float32)
        return X, sfreq

    subj_df = meta_df.groupby("subject_id").agg(y=("y", "first")).reset_index()
    X_dummy = np.zeros((len(subj_df), 1))
    y_subj = subj_df["y"].values.astype(int)
    groups = subj_df["subject_id"].values.astype(str)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_dummy, y_subj, groups), start=1):
        tr_sids = set(subj_df.iloc[tr_idx]["subject_id"].tolist())
        va_sids = set(subj_df.iloc[va_idx]["subject_id"].tolist())

        train_df = meta_df[meta_df["subject_id"].isin(tr_sids)].reset_index(drop=True)
        val_df = meta_df[meta_df["subject_id"].isin(va_sids)].reset_index(drop=True)

        print("\n" + "="*70)
        print(f"[fold{fold:02d}] Train subjects: {len(tr_sids)} | Val subjects: {len(va_sids)}")
        print("="*70)

        model, win_m, subj_m = train_one_fold(
            fold, train_df, val_df, canon, extract_windows_from_raw,
            save_dir=MODEL_SAVE_DIR
        )

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

    win_folds = [r[0] for r in fold_results]
    subj_folds = [r[1] for r in fold_results]
    win_sum = summarize(win_folds)
    subj_sum = summarize(subj_folds)

    print("\n" + "#"*72)
    print("DUAL-BRANCH RESULTS:")
    for i, (w, s) in enumerate(fold_results, start=1):
        print(f"[fold{i:02d}] WIN  bacc={w['bacc']:.4f} f1={w['f1']:.4f} | SUBJ bacc={s['bacc']:.4f} f1={s['f1']:.4f}")

    print("\n" + "#"*72)
    print(f"OVERALL: WIN bacc={win_sum['bacc'][0]:.4f}±{win_sum['bacc'][1]:.4f} | SUBJ bacc={subj_sum['bacc'][0]:.4f}±{subj_sum['bacc'][1]:.4f}")
    print("#"*72)