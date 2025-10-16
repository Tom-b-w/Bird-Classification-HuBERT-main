import os
import librosa
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, AutoFeatureExtractor
from datasets import Dataset as HFDataset, DatasetDict
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_dataset(data_dir):
    data = []
    for species in os.listdir(data_dir):
        sp_path = os.path.join(data_dir, species)
        if os.path.isdir(sp_path):
            for f in os.listdir(sp_path):
                if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    data.append({'path': os.path.join(sp_path, f), 'class_name': species})
    return pd.DataFrame(data)


def get_audio(path, max_len, sr):
    try:
        y, _ = librosa.load(path, sr=sr, duration=4)
        return np.pad(y, (0, max(0, max_len - len(y))))[:max_len]
    except:
        return None


class AudioDataset(Dataset):
    def __init__(self, data):
        self.input_values = torch.FloatTensor(data['input_values'])
        self.attention_mask = torch.LongTensor(data['attention_mask'])
        self.labels = torch.LongTensor(data['label'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'input_values': self.input_values[idx], 'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx]}


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_values, attention_mask):
        pooled = torch.mean(self.wav2vec2(input_values, attention_mask=attention_mask).last_hidden_state, dim=1)
        return self.classifier(pooled)


def run_epoch(model, loader, criterion, optimizer=None):
    model.train() if optimizer else model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.set_grad_enabled(optimizer is not None):
        for batch in tqdm(loader, desc="训练" if optimizer else "验证"):
            input_values, attention_mask, labels = batch['input_values'].to(device), batch['attention_mask'].to(device), \
            batch['labels'].to(device)

            if optimizer:
                optimizer.zero_grad()

            outputs = model(input_values, attention_mask)
            loss = criterion(outputs, labels)

            if optimizer:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100 * correct / total


def main():
    SR, MAX_LEN, BS, EPOCHS = 16000, 64000, 4, 40

    df_train = load_dataset('./dataset/whole_dataset/train_val/train_data_backup')
    df_val = load_dataset('./dataset/whole_dataset/train_val/validation_data_backup')

    cls_label = {s: i for i, s in enumerate(sorted(df_train['class_name'].unique()))}
    for df in [df_train, df_val]:
        df['label'] = df['class_name'].map(cls_label)
        df['audio'] = df['path'].apply(lambda x: get_audio(x, MAX_LEN, SR))
    df_train, df_val = df_train.dropna(subset=['audio']), df_val.dropna(subset=['audio'])

    extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base", return_attention_mask=True)
    preprocess = lambda ex: extractor(ex["audio"], sampling_rate=SR, max_length=MAX_LEN, truncation=True, padding=True,
                                      return_attention_mask=True)

    processed = DatasetDict({
        'train': HFDataset.from_pandas(df_train[['audio', 'label']]),
        'val': HFDataset.from_pandas(df_val[['audio', 'label']])
    }).map(preprocess, remove_columns=["audio"], batched=True)

    train_loader = DataLoader(AudioDataset(processed["train"].shuffle(seed=42).with_format("numpy")[:]), batch_size=BS,
                              shuffle=True)
    val_loader = DataLoader(AudioDataset(processed["val"].shuffle(seed=42).with_format("numpy")[:]), batch_size=BS)

    model = Wav2Vec2Classifier(len(cls_label)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 5)

    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc']}
    best_acc, patience = 0, 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        for k, v in zip(history.keys(), [train_loss, train_acc, val_loss, val_acc]):
            history[k].append(v)

        print(
            f'Epoch {epoch + 1}/{EPOCHS} - Train: {train_loss:.4f}/{train_acc:.2f}% - Val: {val_loss:.4f}/{val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metrics, ylabel in [(ax1, ['train_acc', 'val_acc'], 'Accuracy (%)'),
                                (ax2, ['train_loss', 'val_loss'], 'Loss')]:
        for m in metrics:
            ax.plot(history[m], label=m.split('_')[0])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)

    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump({"num_classes": len(cls_label), "cls_label": cls_label, "best_val_acc": float(best_acc)}, f, indent=2,
                  ensure_ascii=False)

    print(f"训练完成，最佳准确率: {best_acc:.2f}%")
if __name__ == '__main__':
    main()