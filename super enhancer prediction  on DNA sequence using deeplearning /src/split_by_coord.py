import numpy as np
from sklearn.model_selection import train_test_split
import os

os.makedirs("data/processed/human", exist_ok=True)
os.makedirs("data/processed/mouse", exist_ok=True)

def split_species(species):

    X = np.load(f"data/processed/{species}/{species}_X.npy")
    y = np.load(f"data/processed/{species}/{species}_y.npy")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    np.save(f"data/processed/{species}/X_train.npy", X_train)
    np.save(f"data/processed/{species}/X_val.npy", X_val)
    np.save(f"data/processed/{species}/y_train.npy", y_train)
    np.save(f"data/processed/{species}/y_val.npy", y_val)

    print(f"{species} split complete")

split_species("human")
split_species("mouse")