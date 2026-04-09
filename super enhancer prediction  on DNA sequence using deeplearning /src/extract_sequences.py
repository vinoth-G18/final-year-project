from pyfaidx import Fasta
import numpy as np
import os

# limits to avoid laptop crash
MAX_SE = 10000
MAX_TE = 10000


def extract_se(tsv_path, genome_fa, out_prefix, limit=MAX_SE):
    fasta = Fasta(genome_fa, as_raw=True, sequence_always_upper=True)

    seqs = []
    coords = []
    count = 0

    with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().replace('"', '').strip().split("\t")

        chr_i = header.index("se_chr")
        start_i = header.index("se_start")
        end_i = header.index("se_end")

        for line in f:

            if count >= limit:
                break

            parts = line.replace('"', '').strip().split("\t")

            if len(parts) <= max(chr_i, start_i, end_i):
                continue

            try:
                chrom = parts[chr_i]
                start = int(parts[start_i])
                end = int(parts[end_i])
            except:
                continue

            try:
                seq = fasta[chrom][start:end]
            except:
                continue

            seqs.append(seq)
            coords.append((chrom, start, end))

            count += 1

            if count % 1000 == 0:
                print(f"{count} SE sequences processed...")

    np.save(out_prefix + "_seq.npy", np.array(seqs, dtype=object))
    np.save(out_prefix + "_coords.npy", np.array(coords, dtype=object))

    fasta.close()

    print(f"Saved {count} SE from {tsv_path}")


def extract_te(bed_path, genome_fa, out_prefix, limit=MAX_TE):
    fasta = Fasta(genome_fa, as_raw=True, sequence_always_upper=True)

    seqs = []
    coords = []
    count = 0

    with open(bed_path) as f:
        for line in f:

            if count >= limit:
                break

            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split()

            if len(parts) < 3:
                continue

            try:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
            except:
                continue

            try:
                seq = fasta[chrom][start:end]
            except:
                continue

            seqs.append(seq)
            coords.append((chrom, start, end))

            count += 1

            if count % 1000 == 0:
                print(f"{count} TE sequences processed...")

    np.save(out_prefix + "_seq.npy", np.array(seqs, dtype=object))
    np.save(out_prefix + "_coords.npy", np.array(coords, dtype=object))

    fasta.close()

    print(f"Saved {count} TE from {bed_path}")


if __name__ == "__main__":

    os.makedirs("data/processed/human", exist_ok=True)
    os.makedirs("data/processed/mouse", exist_ok=True)

    print("SAFE EXTRACTION STARTED")

    # ===== HUMAN =====
    extract_se(
        "data/raw/human/SE.bed",
        "data/raw/human/genome.fa",
        "data/processed/human/human_SE"
    )

    extract_te(
        "data/raw/human/SE_te.bed",
        "data/raw/human/genome.fa",
        "data/processed/human/human_TE"
    )

    # ===== MOUSE =====
    extract_se(
        "data/raw/mouse/SE_mm.bed",
        "data/raw/mouse/genome.fa",
        "data/processed/mouse/mouse_SE"
    )

    extract_te(
        "data/raw/mouse/SE_te_mm.bed",
        "data/raw/mouse/genome.fa",
        "data/processed/mouse/mouse_TE"
    )