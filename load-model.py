ckpt = torch.load("pretrained/ncf_ds004584.pth", map_location=DEVICE)

model = NeuroConvFormerLite(
    n_ch=len(canon),
    n_time=EXPECTED_WINDOW_SAMPLES
).to(DEVICE)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()
