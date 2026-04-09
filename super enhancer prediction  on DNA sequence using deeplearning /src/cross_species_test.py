import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from model import EnhancerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model_path, X_path, y_path, description):

    X = np.load(X_path)
    y = np.load(y_path)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X, y)

    loader = DataLoader(dataset, batch_size=16)

    model = EnhancerModel().to(device)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device)

            out = model(xb).cpu().numpy()

            preds.extend(out)
            labels.extend(yb.numpy())

    auc = roc_auc_score(labels, preds)

    print(description, "AUC:", auc)


print("Cross Species Experiments\n")

test(
    "checkpoints/finetuned_human_model.pt",
    "data/processed/mouse/X_val.npy",
    "data/processed/mouse/y_val.npy",
    "Human → Mouse"
)

test(
    "checkpoints/finetuned_mouse_model.pt",
    "data/processed/human/X_val.npy",
    "data/processed/human/y_val.npy",
    "Mouse → Human"
)