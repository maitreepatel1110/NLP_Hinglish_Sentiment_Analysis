# NLP_Hinglish_Sentiment_Analysis

# Hinglish Sentiment Classification using Hierarchical DeBERTa-v3

A scalable and robust sentiment classification model designed for code-mixed Hinglish (Hindi-English) text using a custom **Hierarchical DeBERTa-v3** architecture. The model handles long, noisy social media inputs using chunk-wise attention and focuses on class imbalance with Focal Loss.

---

## 🚀 Features

- ✅ Transformer Backbone: [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)
- ✅ Hierarchical attention over chunked sentences
- ✅ Focal Loss for class imbalance
- ✅ Text augmentation using TextAttack
- ✅ Custom Hinglish stopword removal
- ✅ Metrics: Accuracy, Precision, Recall, F1
- ✅ Optimized training loop with early stopping, gradient accumulation, warmup, and weight decay

---

## 📊 Results

| Metric     | Score   |
|------------|---------|
| Accuracy   | 78.76%  |
| Precision  | 78.14%  |
| Recall     | 78.76%  |
| F1 Score   | 78.13%  |

---

## 📁 Dataset

- `FinalTrainingOnly.csv` — CSV with columns `text` and `label`
- `stop_hinglish.txt` — Custom stopword list for Hinglish

Model Overview

The model uses DeBERTa-v3 to embed input chunks, processes them through MultiheadAttention, and classifies the aggregated representation.

Architecture
Base: microsoft/deberta-v3-base
Chunking: Long sequences split into fixed-size chunks (e.g., 64 tokens)
Attention: Multi-head attention across chunk embeddings
Classifier: Dropout → WeightNorm Linear → GELU → LayerNorm → Linear

Training

Train the model using:
model = HierarchicalDeBERTa(...)
trainer(
    model,
    num_epochs,
    optimizer,
    loss_fn,
    lr,
    train_dataloader,
    test_dataloader,
    device
)
Checkpoints are saved as best_model.pth when validation accuracy improves

Evaluation

Evaluate using:
eval(model, test_dataloader, device)
Returns accuracy, precision, recall, and F1-score.

Dependencies

torch
transformers
pandas
scikit-learn
textattack
tqdm

License

This project is released under the MIT License.

Acknowledgements

Microsoft DeBERTa
TextAttack
HuggingFace Transformers
🔗 Connect

📧 maitreepatel57@gmail.com





