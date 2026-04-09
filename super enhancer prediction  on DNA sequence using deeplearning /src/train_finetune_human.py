import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from model import EnhancerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load train data
X_train = np.load("data/processed/human/X_train.npy")
y_train = np.load("data/processed/human/y_train.npy")

# load validation data
X_val = np.load("data/processed/human/X_val.npy")
y_val = np.load("data/processed/human/y_val.npy")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

model = EnhancerModel().to(device)

# load pretrained weights
model.load_state_dict(
    torch.load("checkpoints/pretrain_base_model.pt")
)

print("Loaded pretrained model")

# freeze CNN layers
for name, param in model.named_parameters():
    if "conv" in name:
        param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5
)

loss_fn = torch.nn.BCELoss()

epochs = 20
best_auc = 0

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        preds = model(xb)

        loss = loss_fn(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # validation
    model.eval()

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in val_loader:

            xb = xb.to(device)

            preds = model(xb).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    auc = roc_auc_score(all_labels, all_preds)

    print(f"Epoch {epoch+1} loss {avg_loss:.4f} val_auc {auc:.4f}")

    # save best model
    if auc > best_auc:

        best_auc = auc

        torch.save(
            model.state_dict(),
            "checkpoints/finetuned_human_model.pt"
        )

        print("New best model saved")

print("Best Human AUC:", best_auc)