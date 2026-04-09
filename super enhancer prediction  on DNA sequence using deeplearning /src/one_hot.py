import numpy as np

MAP = {
    "A": [1,0,0,0],
    "C": [0,1,0,0],
    "G": [0,0,1,0],
    "T": [0,0,0,1],
    "N": [0,0,0,0]
}

def one_hot(seqs):
    X = np.zeros((len(seqs), 3000, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, b in enumerate(s):
            X[i, j] = MAP.get(b, [0,0,0,0])
    return X


def build(se_path, te_path, out_prefix):
    se = np.load(se_path, allow_pickle=True)
    te = np.load(te_path, allow_pickle=True)

    X = np.concatenate([one_hot(se), one_hot(te)])
    y = np.concatenate([np.ones(len(se)), np.zeros(len(te))])

    np.save(out_prefix + "_X.npy", X)
    np.save(out_prefix + "_y.npy", y)


if __name__ == "__main__":
    build("data/processed/human/human_SE_seq.npy",
          "data/processed/human/human_TE_seq.npy",
          "data/processed/human/human")

    build("data/processed/mouse/mouse_SE_seq.npy",
          "data/processed/mouse/mouse_TE_seq.npy",
          "data/processed/mouse/mouse")
