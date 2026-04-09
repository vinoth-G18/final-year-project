import torch
import numpy as np
from model import EnhancerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# One-hot encoding
# -----------------------
def one_hot_encode(seq):

    mapping = {
        "A":[1,0,0,0],
        "C":[0,1,0,0],
        "G":[0,0,1,0],
        "T":[0,0,0,1]
    }

    encoded = []

    for base in seq:
        encoded.append(mapping.get(base,[0,0,0,0]))

    return np.array(encoded)


# -----------------------
# Sequence statistics
# -----------------------
def gc_content(seq):

    gc = seq.count("G") + seq.count("C")

    return gc / len(seq)


# -----------------------
# Motif density estimation
# -----------------------
def motif_density(seq):

    motifs = ["GGAA","TTCC","CGCG","GATA","TATA"]

    count = 0

    for m in motifs:
        count += seq.count(m)

    density = count / len(seq)

    if density > 0.02:
        return "High"
    elif density > 0.01:
        return "Moderate"
    else:
        return "Low"


# -----------------------
# Important regions finder
# -----------------------
def find_important_regions(seq):

    window = 50
    step = 50

    regions = []

    for i in range(0, len(seq)-window, step):

        sub = seq[i:i+window]

        score = (sub.count("G") + sub.count("C")) / window

        if score > 0.6:
            regions.append((i, i+window))

    return regions[:3]


# -----------------------
# Conservation estimation
# -----------------------
def conservation_score(gc):

    if gc > 0.55:
        return "High"
    elif gc > 0.45:
        return "Moderate"
    else:
        return "Low"


# -----------------------
# Load model
# -----------------------
model = EnhancerModel().to(device)

model.load_state_dict(
    torch.load("checkpoints/finetuned_human_model.pt")
)

model.eval()


# -----------------------
# Main analysis
# -----------------------
def analyze_sequence(seq):

    seq = seq.upper()

    encoded = one_hot_encode(seq)

    X = torch.tensor(encoded,dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(X).item()

    prediction = "Super Enhancer" if prob > 0.5 else "Typical Enhancer"

    if prob > 0.9:
        category = "Strong Super Enhancer"
    elif prob > 0.75:
        category = "Moderate Super Enhancer"
    else:
        category = "Weak Regulatory Region"

    gc = gc_content(seq)

    motifs = motif_density(seq)

    conservation = conservation_score(gc)

    regions = find_important_regions(seq)

    print("\nPrediction Result")
    print("-----------------")
    print("Prediction:", prediction)
    print("Confidence Score:", round(prob,2))
    print("Category:", category)

    print("\nSequence Information")
    print("--------------------")
    print("Length:",len(seq),"bp")
    print("GC Content:", round(gc*100,1),"%")

    print("\nRegulatory Analysis")
    print("-------------------")

    print("Important Regions:")

    for r in regions:
        print(f"  {r[0]}–{r[1]}")

    print("\nMotif Density:", motifs)

    print("Cross-Species Conservation:", conservation)


# -----------------------
# Example test
# -----------------------
if __name__ == "__main__":

    example_seq = "ATCGGCGGATCGGCGGATCGGCGGATCG"*120

    analyze_sequence(example_seq)