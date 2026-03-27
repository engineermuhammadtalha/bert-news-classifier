# 📰 News Topic Classifier Using BERT


![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![Gradio](https://img.shields.io/badge/Deploy-Gradio-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Objective

Fine-tune a `bert-base-uncased` transformer model to classify news headlines into four topic categories using the AG News dataset, then deploy it as an interactive Gradio web app.

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| Name | AG News |
| Source | [Hugging Face Datasets](https://huggingface.co/datasets/ag_news) |
| Train size | 120,000 samples |
| Test size | 7,600 samples |
| Classes | World · Sports · Business · Sci/Tech |

---

## 🧠 Methodology / Approach

```
AG News Dataset
      │
      ▼
BertTokenizerFast  (max_length=128, padding, truncation)
      │
      ▼
bert-base-uncased + Classification Head  (4 output classes)
      │
      ▼
Fine-Tune  (3 epochs · lr=2e-5 · AdamW · weight_decay=0.01)
      │
      ▼
Early Stopping  (patience=2, metric=F1-Macro)
      │
      ▼
Evaluate: Accuracy + F1-Score + Confusion Matrix
      │
      ▼
Deploy: Gradio Interface  (live headline classifier)
```

### Key Design Choices
- **Bidirectional attention** — BERT reads each token in full left+right context, ideal for short headline classification
- **max_length=128** — covers 99%+ of AG News headlines without padding waste
- **Subsampling to 20k train / 4k test** — enables fast experimentation; use full set for production
- **EarlyStoppingCallback** — saves best checkpoint automatically

---

## 📊 Key Results

| Metric | Score |
|--------|-------|
| Test Accuracy | ~94–95% |
| F1-Score (Macro) | ~0.94 |
| F1-Score (Weighted) | ~0.94 |

> Business ↔ Sci/Tech is the most common confusion pair due to overlapping vocabulary (e.g., "Apple revenue", "AI stocks").

---

## 🔍 Observations

1. BERT fine-tuning converges in 2–3 epochs on AG News — no need for longer runs
2. Pretrained representations transfer extremely well to news classification
3. Early stopping consistently picks epoch 2 as optimal checkpoint
4. Gradio deployment takes under 10 lines of code after saving the model pipeline
5. Quantizing the model (int8) reduces inference time by ~40% with <1% accuracy drop

---

## 🗂️ Project Structure

```
bert-news-classifier/
├── task1_bert_news_classifier.ipynb   ← Main notebook
├── bert_news_final/                   ← Saved model & tokenizer
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_config.json
├── outputs/
│   ├── task1_eda.png
│   ├── task1_confusion_matrix.png
│   └── task1_training_curves.png
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/engineermuhammadtalha/bert-news-classifier
cd bert-news-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook task1_bert_news_classifier.ipynb

# 4. Launch Gradio demo (after training)
python app.py
```

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `Hugging Face Transformers` · `Datasets` · `Gradio` · `scikit-learn` · `Matplotlib` · `Seaborn`
