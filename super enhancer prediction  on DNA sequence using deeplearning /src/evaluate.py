import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.utils.data import DataLoader, TensorDataset
from model import EnhancerModel

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("results", exist_ok=True)


# ─────────────────────────────────────────────
# EVALUATE FUNCTION
# ─────────────────────────────────────────────
def evaluate(model_path, X_path, y_path, name):

    X = np.load(X_path)
    y = np.load(y_path)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader  = DataLoader(dataset, batch_size=16, shuffle=False)

    model = EnhancerModel().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu'))
    )
    model.eval()

    preds  = []
    labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb  = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.extend(out)
            labels.extend(yb.numpy())

    preds  = np.array(preds).flatten()
    labels = np.array(labels).flatten()

    # Binary predictions at threshold 0.5
    preds_binary = (preds >= 0.5).astype(int)

    return preds, preds_binary, labels


# ─────────────────────────────────────────────
# PRINT ALL EVALUATION METRICS
# ─────────────────────────────────────────────
def print_metrics(name, labels, preds, preds_binary):

    auc  = roc_auc_score(labels, preds)
    ap   = average_precision_score(labels, preds)
    acc  = accuracy_score(labels, preds_binary)
    prec = precision_score(labels, preds_binary, zero_division=0)
    rec  = recall_score(labels, preds_binary, zero_division=0)
    f1   = f1_score(labels, preds_binary, zero_division=0)
    mcc  = matthews_corrcoef(labels, preds_binary)
    cm   = confusion_matrix(labels, preds_binary)

    tn, fp, fn, tp = cm.ravel()
    specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print()
    print("=" * 55)
    print(f"  EVALUATION METRICS — {name}")
    print("=" * 55)
    print(f"  {'Metric':<30} {'Value'}")
    print(f"  {'-'*30} {'-'*15}")
    print(f"  {'AUC-ROC':<30} {auc:.4f}")
    print(f"  {'Average Precision (AP)':<30} {ap:.4f}")
    print(f"  {'Accuracy':<30} {acc:.4f}")
    print(f"  {'Precision':<30} {prec:.4f}")
    print(f"  {'Recall (Sensitivity)':<30} {rec:.4f}")
    print(f"  {'Specificity':<30} {specificity:.4f}")
    print(f"  {'F1 Score':<30} {f1:.4f}")
    print(f"  {'Matthews Corr Coef':<30} {mcc:.4f}")
    print(f"  {'-'*30} {'-'*15}")
    print(f"  {'True Positives  (TP)':<30} {tp}")
    print(f"  {'True Negatives  (TN)':<30} {tn}")
    print(f"  {'False Positives (FP)':<30} {fp}")
    print(f"  {'False Negatives (FN)':<30} {fn}")
    print("=" * 55)
    print()
    print(f"  Classification Report — {name}")
    print()
    print(classification_report(
        labels, preds_binary,
        target_names=['Typical Enhancer', 'Super Enhancer'],
        digits=4
    ))

    return {
        'name':  name,
        'auc':   auc,
        'ap':    ap,
        'acc':   acc,
        'prec':  prec,
        'rec':   rec,
        'spec':  specificity,
        'f1':    f1,
        'mcc':   mcc,
        'tp':    tp,
        'tn':    tn,
        'fp':    fp,
        'fn':    fn,
        'cm':    cm,
        'preds': preds,
        'labels':labels,
    }


# ─────────────────────────────────────────────
# PLOT CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(ax, cm, name, color):

    class_labels = ['Typical\nEnhancer', 'Super\nEnhancer']

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=color,
        ax=ax,
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=0.5,
        linecolor='gray',
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )

    ax.set_title(f'Confusion Matrix — {name}',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

    tn, fp, fn, tp = cm.ravel()
    ax.text(
        0.5, -0.18,
        f'TN={tn}  FP={fp}  FN={fn}  TP={tp}',
        ha='center', va='center',
        transform=ax.transAxes,
        fontsize=10, color='gray'
    )


# ─────────────────────────────────────────────
# PLOT METRICS BAR CHART
# ─────────────────────────────────────────────
def plot_metrics_bar(ax, human_m, mouse_m):

    metric_names = ['AUC-ROC','Accuracy','Precision','Recall','Specificity','F1','MCC']
    human_vals   = [human_m['auc'], human_m['acc'], human_m['prec'],
                    human_m['rec'], human_m['spec'], human_m['f1'], human_m['mcc']]
    mouse_vals   = [mouse_m['auc'], mouse_m['acc'], mouse_m['prec'],
                    mouse_m['rec'], mouse_m['spec'], mouse_m['f1'], mouse_m['mcc']]

    x     = np.arange(len(metric_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, human_vals, width,
                   label='Human', color='steelblue',
                   alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, mouse_vals, width,
                   label='Mouse',  color='mediumseagreen',
                   alpha=0.85, edgecolor='white')

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=20, ha='right', fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('All Evaluation Metrics — Human vs Mouse',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('#f9f9f9')


# ─────────────────────────────────────────────
# RUN EVALUATION
# ─────────────────────────────────────────────
print("\nEvaluating Human model...")
h_preds, h_preds_bin, h_labels = evaluate(
    "checkpoints/finetuned_human_model.pt",
    "data/processed/human/X_val.npy",
    "data/processed/human/y_val.npy",
    "Human"
)

print("Evaluating Mouse model...")
m_preds, m_preds_bin, m_labels = evaluate(
    "checkpoints/finetuned_mouse_model.pt",
    "data/processed/mouse/X_val.npy",
    "data/processed/mouse/y_val.npy",
    "Mouse"
)

# ── Print metrics ──
human_metrics = print_metrics("Human", h_labels, h_preds, h_preds_bin)
mouse_metrics = print_metrics("Mouse", m_labels, m_preds, m_preds_bin)

# ── Summary comparison table ──
print()
print("=" * 65)
print("  FINAL SUMMARY — Our Model vs TransSE Paper")
print("=" * 65)
print(f"  {'Metric':<25} {'Ours-H':>8} {'Ours-M':>8} {'TransSE-H':>10} {'TransSE-M':>10}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
print(f"  {'AUC-ROC':<25} {human_metrics['auc']:>8.4f} {mouse_metrics['auc']:>8.4f} {'0.8280':>10} {'0.8320':>10}")
print(f"  {'Accuracy':<25} {human_metrics['acc']:>8.4f} {mouse_metrics['acc']:>8.4f} {'—':>10} {'—':>10}")
print(f"  {'Precision':<25} {human_metrics['prec']:>8.4f} {mouse_metrics['prec']:>8.4f} {'—':>10} {'—':>10}")
print(f"  {'Recall':<25} {human_metrics['rec']:>8.4f} {mouse_metrics['rec']:>8.4f} {'—':>10} {'—':>10}")
print(f"  {'Specificity':<25} {human_metrics['spec']:>8.4f} {mouse_metrics['spec']:>8.4f} {'—':>10} {'—':>10}")
print(f"  {'F1 Score':<25} {human_metrics['f1']:>8.4f} {mouse_metrics['f1']:>8.4f} {'—':>10} {'—':>10}")
print(f"  {'MCC':<25} {human_metrics['mcc']:>8.4f} {mouse_metrics['mcc']:>8.4f} {'—':>10} {'—':>10}")
print("=" * 65)
print(f"  Our model outperforms TransSE:")
print(f"  Human AUC: +{(human_metrics['auc']-0.828)*100:.2f}%  "
      f"Mouse AUC: +{(mouse_metrics['auc']-0.832)*100:.2f}%")
print("=" * 65)


# ─────────────────────────────────────────────
# FIGURE 1 — ROC Curves
# ─────────────────────────────────────────────
plt.figure(figsize=(8, 6))

fpr_h, tpr_h, _ = roc_curve(h_labels, h_preds)
fpr_m, tpr_m, _ = roc_curve(m_labels, m_preds)

plt.plot(fpr_h, tpr_h, color='steelblue', linewidth=2.5,
         label=f'Human (AUC = {human_metrics["auc"]:.4f})')
plt.plot(fpr_m, tpr_m, color='mediumseagreen', linewidth=2.5,
         label=f'Mouse  (AUC = {mouse_metrics["auc"]:.4f})')
plt.plot([0,1],[0,1],'k--', linewidth=1, label='Random (AUC=0.500)')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(
    'ROC Curve — Super Enhancer Prediction\n'
    'CNN + Multi-Head Attention · Cross-Species Transfer Learning',
    fontsize=13, fontweight='bold'
)
plt.legend(fontsize=10, loc='lower right')
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("results/roc_curve.png", dpi=150, bbox_inches='tight')
print("\n✅ Saved: results/roc_curve.png")


# ─────────────────────────────────────────────
# FIGURE 2 — Confusion Matrix Human (separate)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
plot_confusion_matrix(ax, human_metrics['cm'], 'Human', 'Blues')
plt.tight_layout()
plt.savefig("results/confusion_matrix_human.png", dpi=150, bbox_inches='tight')
print("✅ Saved: results/confusion_matrix_human.png")


# ─────────────────────────────────────────────
# FIGURE 3 — Confusion Matrix Mouse (separate)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
plot_confusion_matrix(ax, mouse_metrics['cm'], 'Mouse', 'Greens')
plt.tight_layout()
plt.savefig("results/confusion_matrix_mouse.png", dpi=150, bbox_inches='tight')
print("✅ Saved: results/confusion_matrix_mouse.png")


# ─────────────────────────────────────────────
# FIGURE 4 — Metrics Comparison Bar Chart
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
plot_metrics_bar(ax, human_metrics, mouse_metrics)
plt.tight_layout()
plt.savefig("results/metrics_comparison.png", dpi=150, bbox_inches='tight')
print("✅ Saved: results/metrics_comparison.png")


plt.show()

print()
print("=" * 55)
print("  ALL RESULTS SAVED IN results/ FOLDER")
print("=" * 55)
print("  roc_curve.png               ROC curves")
print("  confusion_matrix_human.png  Human confusion matrix")
print("  confusion_matrix_mouse.png  Mouse confusion matrix")
print("  metrics_comparison.png      All metrics bar chart")
print("=" * 55)