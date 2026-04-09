# 🧬 Super Enhancer Prediction using Deep Learning

A deep learning-based system to classify DNA sequences as **Super Enhancers (SE)** or **Typical Enhancers (TE)** using **CNN + Multi-Head Attention** with **Cross-Species Transfer Learning**.

---

## 🚀 Overview

Enhancers are DNA regulatory elements that control gene expression.  
**Super Enhancers (SEs)** are clusters of enhancers that strongly regulate important genes, especially in diseases like cancer.

Traditional identification methods such as **ChIP-seq** are expensive and time-consuming.  
This project provides an **AI-based alternative** for fast and accurate prediction using only DNA sequence data.

---

## 🎯 Key Features

- 🔬 Predicts **Super Enhancer vs Typical Enhancer**
- 🧠 Uses **CNN + Attention architecture**
- 🌍 Implements **Cross-Species Transfer Learning (Human + Mouse)**
- 📊 Provides **ROC-AUC evaluation**
- 🧪 Includes **biological interpretation**:
  - GC Content
  - Motif Density
  - Important Regions
- 🌐 Interactive **Streamlit Web App**

---

## 🏗️ Project Pipeline
Sequence Extraction
↓
Padding / Trimming (3000 bp)
↓
One-Hot Encoding
↓
Train / Validation Split
↓
Pretraining (Human + Mouse)
↓
Fine-tuning (Human & Mouse)
↓
Evaluation
↓
Prediction + Biological Analysis


---

## 🧠 Model Architecture

- Input: `(3000, 4)` one-hot encoded DNA sequence  
- Conv1D layers for motif detection  
- MaxPooling for dimensionality reduction  
- Multi-Head Self Attention for long-range dependencies  
- Global Average Pooling  
- Dense + Dropout  
- Sigmoid Output (Binary classification)

---

## 📊 Results

| Dataset | AUC Score |
|--------|----------|
| Human  | **0.849** |
| Mouse  | **0.912** |

✔ Outperforms baseline **TransSE model**

---

## 🧬 Dataset

- Species: Human & Mouse  
- Classes: Super Enhancer (SE), Typical Enhancer (TE)  
- Sequence length: 3000 bp  
- Encoding: One-hot  

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/SuperEnhancer-AI.git
cd SuperEnhancer-AI
pip install -r requirements.txt

▶️ Run Web Application
py -3.10 -m streamlit run webapp/app.py
http://localhost:8501

📈 Evaluation Metrics
Accuracy
Precision
Recall
F1-score
ROC Curve
AUC Score
