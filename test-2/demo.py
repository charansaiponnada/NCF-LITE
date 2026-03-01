import os
import scipy.io as sio
import mne
import numpy as np

MAT_DIR = r"C:\projects\NCF-lite\dataset\PD REST"
EDF_DIR = r"C:\projects\NCF-lite\dataset\PD REST\edf"

os.makedirs(EDF_DIR, exist_ok=True)


def load_mat(fp):
    mat = sio.loadmat(fp, squeeze_me=True, struct_as_record=False)
    EEG = mat["EEG"]

    data = EEG.data
    if data.ndim == 3:
        data = data[0]
    if data.shape[0] > data.shape[1]:
        data = data.T

    sfreq = float(EEG.srate)
    ch_names = [c.labels.strip() for c in EEG.chanlocs]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data.astype(np.float32), info, verbose=False)
    return raw

for fname in os.listdir(MAT_DIR):
    if not fname.endswith(".mat"):
        continue

    mat_path = os.path.join(MAT_DIR, fname)
    raw = load_mat(mat_path)

    edf_name = fname.replace(".mat", ".edf")
    out_path = os.path.join(EDF_DIR, edf_name)

    mne.export.export_raw(out_path, raw, fmt="edf", physical_range=(-200e-6, 200e-6))
    print("Saved", out_path)
