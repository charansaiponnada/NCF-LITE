import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# NeuroConvFormer-Lite (MODEL ONLY)
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

class NeuroConvFormerLite(nn.Module):
    def __init__(self, n_ch, d_model=64):
        super().__init__()

        ks = [7, 15, 31]
        self.dw_convs = nn.ModuleList([
            nn.Conv1d(n_ch, n_ch, k, padding=k//2, groups=n_ch, bias=False)
            for k in ks
        ])
        self.pw_mix = nn.Conv1d(n_ch * len(ks), d_model, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(d_model)

        self.se = SEBlock(d_model)
        self.ds = nn.Conv1d(d_model, d_model, 5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(d_model)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=2*d_model,
            batch_first=True,
            norm_first=True
        )
        self.tr = nn.TransformerEncoder(enc, num_layers=2)

        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        feats = [dw(x) for dw in self.dw_convs]
        x = torch.cat(feats, dim=1)
        x = F.gelu(self.bn1(self.pw_mix(x)))
        x = self.se(x)
        x = F.gelu(self.bn2(self.ds(x)))
        x = x.transpose(1, 2)
        x = self.tr(x)
        w = torch.softmax(self.attn(x), dim=1)
        x = (x * w).sum(dim=1)
        return self.head(x)
