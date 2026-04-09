import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import EnhancerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Load datasets
# ---------------------------

human_X = np.load("data/processed/human/human_X.npy")
human_y = np.load("data/processed/human/human_y.npy")

mouse_X = np.load("data/processed/mouse/mouse_X.npy")
mouse_y = np.load("data/processed/mouse/mouse_y.npy")

X = np.concatenate([human_X, mouse_X])
y = np.concatenate([human_y, mouse_y])

print("Total samples:", len(X))

# ---------------------------
# Convert to tensors
# ---------------------------

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

# ---------------------------
# Model
# ---------------------------

model = EnhancerModel().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-5
)

loss_fn = torch.nn.BCELoss()

epochs = 20

# ---------------------------
# Training
# ---------------------------

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for xb, yb in loader:

        xb = xb.to(device)
        yb = yb.to(device)

        preds = model(xb)

        loss = loss_fn(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(f"Epoch {epoch+1}/{epochs} loss {avg_loss:.4f}")

# ---------------------------
# Save model
# ---------------------------

torch.save(
    model.state_dict(),
    "checkpoints/pretrain_base_model.pt"
)

print("Pretraining complete")