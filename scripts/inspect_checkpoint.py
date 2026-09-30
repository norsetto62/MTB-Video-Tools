import torch

path = "output/mtb_temporal_cnn.pt"

checkpoint = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)

print("Checkpoint keys:")
for key in checkpoint:
    print(f"  {key}")

print()
print(f"Model tensors:   {len(checkpoint['model_state_dict'])}")
print(f"Features:        {checkpoint['n_features']}")
print(f"Labels:          {checkpoint['n_labels']}")
print(f"Label names:     {checkpoint['labels']}")
print(f"Feature names:   {len(checkpoint['feature_names'])}")
print(f"Mean shape:      {getattr(checkpoint['normalization_mean'], 'shape', None)}")
print(f"Std shape:       {getattr(checkpoint['normalization_std'], 'shape', None)}")

history = checkpoint["history"]

print(f"History entries: {len(history)}")

best = min(history, key=lambda x: x["val_loss"])

print()
print(f"Best epoch:      {best['epoch']}")
print(f"Best val_loss:   {best['val_loss']:.4f}")

print()
print("Last history entry:")
print(history[-1])