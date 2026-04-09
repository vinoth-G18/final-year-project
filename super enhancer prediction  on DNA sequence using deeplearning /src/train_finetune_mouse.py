import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import EnhancerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

X_train = np.load("data/processed/mouse/X_train.npy")
y_train = np.load("data/processed/mouse/y_train.npy")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_train,y_train)
loader = DataLoader(dataset,batch_size=64,shuffle=True)

model = EnhancerModel().to(device)

model.load_state_dict(
    torch.load("checkpoints/pretrain_base_model.pt")
)

optimizer = torch.optim.Adam(model.parameters(),lr=2e-5)
loss_fn = torch.nn.BCELoss()

epochs = 8

for epoch in range(epochs):

    total_loss = 0

    for xb,yb in loader:

        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)

        loss = loss_fn(pred,yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} loss {total_loss/len(loader):.4f}")

torch.save(model.state_dict(),
           "checkpoints/finetuned_mouse_model.pt")

print("Mouse fine-tuning complete")