import numpy as np

TARGET_LEN = 3000

def pad_or_trim(seq):
    L = len(seq)
    if L == TARGET_LEN:
        return seq
    if L < TARGET_LEN:
        pad = TARGET_LEN - L
        left = pad // 2
        right = pad - left
        return "N"*left + seq + "N"*right
    else:
        start = (L - TARGET_LEN) // 2
        return seq[start:start+TARGET_LEN]


def process(in_path, out_path):
    seqs = np.load(in_path, allow_pickle=True)
    new_seqs = [pad_or_trim(s) for s in seqs]
    np.save(out_path, np.array(new_seqs, dtype=object))


if __name__ == "__main__":
    process("data/processed/human/human_SE_seq.npy",
            "data/processed/human/human_SE_seq.npy")
    process("data/processed/human/human_TE_seq.npy",
            "data/processed/human/human_TE_seq.npy")

    process("data/processed/mouse/mouse_SE_seq.npy",
            "data/processed/mouse/mouse_SE_seq.npy")
    process("data/processed/mouse/mouse_TE_seq.npy",
            "data/processed/mouse/mouse_TE_seq.npy")
