# Colab & Data Setup
from google.colab import drive
import os
import sys
from sklearn.utils import shuffle
import pandas as pd

# Mount Google Drive
drive.mount('/content/gdrive')

# Set data path
drive_data_dir = "/content/gdrive/My Drive/Research/"
filename = "FinalTrainingOnly.csv"
data_path = os.path.join(drive_data_dir, filename)
stopwords_path = os.path.join(drive_data_dir, "stop_hinglish.txt")

# Check files
if not os.path.exists(data_path):
    print(f"Error: File not found at {data_path}.")
    sys.exit(1)
if not os.path.exists(stopwords_path):
    print(f"Error: Stop words file not found at {stopwords_path}.")
    sys.exit(1)

# Load stop words
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stop_words = set(line.strip().lower() for line in f if line.strip())
print(f"Loaded {len(stop_words)} stop words.")

# PyTorch and Transformers Setup
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler, AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np

# Define device (ADDED HERE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# TextAttack augmentation (optional, for oversampling)
try:
    from textattack.augmentation import EasyDataAugmenter
    augmenter = EasyDataAugmenter(pct_words_to_swap=0.1)
except:
    print("Warning: textattack not installed. Oversampling with augmentation will be skipped.")
    augmenter = None

def remove_stopwords(text, stop_words):
    words = str(text).split()
    filtered = [w for w in words if w.lower() not in stop_words]
    return ' '.join(filtered)

def augment_text(text, augmenter, n=1):
    if augmenter is None:
        return [text]
    return [augmenter.augment(text)[0] for _ in range(n)]

class HinglishDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, stop_words=None, augmenter=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stop_words = stop_words
        self.augmenter = augmenter

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.stop_words:
            text = remove_stopwords(text, self.stop_words)
        label = self.labels[idx]
        if self.augmenter and np.random.rand() < 0.5:  # 50% chance to augment
            text = augment_text(text, self.augmenter, 1)[0]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }

def read_data(path, batch_size, tokenizer, max_len, stop_words, augmenter=None):
    df = pd.read_csv(path, header=None)
    df = df.drop(columns=[0])
    df.columns = ['text', 'label']
    print(f"Loaded dataset with {len(df)} rows.")

    df = shuffle(df).reset_index(drop=True)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    # Map labels 1,2 to 0,1 (for CrossEntropyLoss compatibility)
    label_map = {1: 0, 2: 1}
    labels = [label_map[l] if l in label_map else l for l in labels]
    unique_labels = np.unique(labels)
    print(f"Labels after mapping: {unique_labels}")
    class_weights = torch.tensor([0.6, 0.4], dtype=torch.float).to(device)  # Custom weights, or use compute_class_weight
    loss_fn = FocalLoss(alpha=class_weights, gamma=2)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    train_dataset = HinglishDataset(train_texts, train_labels, tokenizer, max_len, stop_words, augmenter)
    test_dataset = HinglishDataset(test_texts, test_labels, tokenizer, max_len, stop_words)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader, loss_fn

# -------- HIERARCHICAL DeBERTa-v3 MODEL --------
class HierarchicalDeBERTa(nn.Module):
    def __init__(self, model_name, chunk_size=64, num_labels=2, dropout_rate=0.3):  # Added dropout_rate
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.chunk_size = chunk_size
        self.hidden_size = self.bert.config.hidden_size
        self.num_labels = num_labels
        self.chunk_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=dropout_rate,  # Use dropout_rate here
            batch_first=True
        )
        # Advanced regularization: weight norm and dropout
        self.classifier = nn.Sequential(
            weight_norm(nn.Linear(self.hidden_size, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),  # Use dropout_rate here
            weight_norm(nn.Linear(256, num_labels)),
            nn.Dropout(dropout_rate)   # Use dropout_rate here
        )

    def chunk_sequence(self, hidden_states, attention_mask):
        batch_size, seq_len, hidden_dim = hidden_states.size()
        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            hidden_states = torch.cat(
                [hidden_states, torch.zeros(batch_size, pad_len, hidden_dim, device=hidden_states.device)], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.zeros(batch_size, pad_len, device=attention_mask.device)], dim=1)
        num_chunks = hidden_states.size(1) // self.chunk_size
        chunks = hidden_states.view(batch_size, num_chunks, self.chunk_size, hidden_dim)
        chunk_mask = attention_mask.view(batch_size, num_chunks, self.chunk_size)
        return chunks, chunk_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        chunks, chunk_mask = self.chunk_sequence(hidden_states, attention_mask)
        chunk_vectors = chunks.mean(dim=2)  # (batch, num_chunks, hidden)
        # Hierarchical attention across chunks
        attn_output, _ = self.chunk_attention(
            chunk_vectors, chunk_vectors, chunk_vectors,
            key_padding_mask=~chunk_mask.any(dim=2).bool()
        )
        pooled = attn_output.mean(dim=1)  # (batch, hidden)
        logits = self.classifier(pooled)
        return logits

# Custom scheduler for triangular learning rate
from math import floor
class TriangularLR:
    def __init__(self, optimizer, max_lr, min_lr, cycle_len, steps_per_epoch):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_len = cycle_len
        self.steps_per_epoch = steps_per_epoch
        self.step_count = 0

    def step(self):
        cycle = floor(self.step_count / (2 * self.cycle_len * self.steps_per_epoch))
        x = abs(self.step_count / (self.cycle_len * self.steps_per_epoch) - 2 * cycle - 1)
        lr = self.min_lr + (self.max_lr - self.min_lr) * (1 - x)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.step_count += 1

# Trainer with early stopping, decoupled weight decay, and model saving
def trainer(model, epochs, opt, loss_fn, lr, train_dataloader, test_dataloader, device, grad_accum=2, weight_decay=0.01):  # Removed cycle_len
    optimizer = opt(model.parameters(), lr=lr, weight_decay=weight_decay)  # AdamW with decoupled weight decay
    num_training_steps = epochs * len(train_dataloader) // grad_accum
    scheduler = get_scheduler(
        "linear",  # Use "linear" scheduler
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),  # 10% warmup
        num_training_steps=num_training_steps
    )
    model.to(device)
    best_val_loss = float('inf')
    patience = 2  # Reduce patience
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss = loss / grad_accum  # Scale loss for gradient accumulation
            loss.backward()
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step() # Step the linear scheduler here!
                optimizer.zero_grad()
            total_loss += loss.item() * grad_accum  # Scale back up for logging
        tqdm.write(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss / len(train_dataloader):.4f}")

        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                val_loss += loss_fn(logits, labels).item()
        val_loss /= len(test_dataloader)
        tqdm.write(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    return model

def eval(model, test_dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    model.to(device)
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        unique_labels = np.unique(all_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', labels=unique_labels)
    except ValueError:
        print("Warning: Not all classes present in evaluation set. Using micro average for precision/recall/f1.")
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
    print(f"\n Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# ---- MAIN SETTINGS ----
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
lr = 1e-5                       # Increased learning rate
batch_size = 8                 
grad_accum = 2                   # Keep as is
epochs = 10                      # Increased epochs (early stopping will prevent overfitting)
max_len = 256
opt = optim.AdamW
dropout_rate = 0.3              # Increased dropout

# Clear GPU cache
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Load data with stop words, augmentation, and Focal Loss
train_dataloader, test_dataloader, loss_fn = read_data(data_path, batch_size, tokenizer, max_len, stop_words, augmenter)

# --- Train Hierarchical DeBERTa-v3 model ---
print("Starting Hierarchical DeBERTa-v3 model training...")
torch.cuda.empty_cache()
model = HierarchicalDeBERTa(model_name=model_name, chunk_size=64, num_labels=2, dropout_rate=dropout_rate).to(device)  # Pass dropout_rate
print("Model loaded.")

model = trainer(
    model,
    epochs,
    opt,
    loss_fn,
    lr,
    train_dataloader,
    test_dataloader,
    device,
    grad_accum=grad_accum,
    weight_decay=0.01            # Added weight decay
)
print("Model training finished.")

# --- Evaluate ---
print("Evaluating Hierarchical DeBERTa-v3 model...")
model.to(device)
eval(model, test_dataloader, device)
print("Model evaluation finished.")


'''
 Accuracy: 0.7876
Precision: 0.7814
Recall: 0.7876
F1 Score: 0.7813
'''