import os
import librosa
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import json
import matplotlib.pyplot as plt
from transformers import HubertModel, AutoFeatureExtractor
from datasets import Dataset as HFDataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = torch.cuda.is_available()
if use_amp:
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()


def load_dataset(data_dir):
    return pd.DataFrame([{'path': os.path.join(sp_path, f), 'class_name': species}
                         for species in os.listdir(data_dir)
                         for sp_path in [os.path.join(data_dir, species)]
                         if os.path.isdir(sp_path)
                         for f in os.listdir(sp_path)
                         if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')) and os.path.getsize(
            os.path.join(sp_path, f)) > 0])


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
        self.labels = torch.LongTensor(data['class_label'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'input_values': self.input_values[idx], 'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx]}


class HubertForAudioClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        for param in self.hubert.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.conv_params = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]

    def forward(self, input_values, attention_mask=None):
        hidden_states = self.hubert(input_values, attention_mask=attention_mask).last_hidden_state

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1)
            for k, s in self.conv_params:
                lengths = (lengths - k) // s + 1
            batch_size, seq_len = hidden_states.shape[:2]
            mask = torch.zeros((batch_size, seq_len), device=hidden_states.device, dtype=torch.bool)
            for i, l in enumerate(lengths):
                mask[i, :min(l.item(), seq_len)] = True
            mask = mask.unsqueeze(-1).expand_as(hidden_states).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden_states.mean(dim=1)

        return self.classifier(pooled)


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    model.train() if optimizer else model.eval()
    total_loss, correct, total, preds, labels = 0, 0, 0, [], []

    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            iv, am, lb = batch['input_values'].to(device), batch['attention_mask'].to(device), batch['labels'].to(
                device)

            if optimizer:
                optimizer.zero_grad()
                if scaler:
                    with autocast():
                        outputs = model(iv, am)
                        loss = criterion(outputs, lb)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(iv, am)
                    loss = criterion(outputs, lb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            else:
                outputs = model(iv, am)
                loss = criterion(outputs, lb)

            total_loss += loss.item()
            pred = outputs.argmax(1)
            correct += (pred == lb).sum().item()
            total += lb.size(0)
            if not optimizer:
                preds.extend(pred.cpu().numpy())
                labels.extend(lb.cpu().numpy())

    return (total_loss / len(loader), 100 * correct / total, preds, labels) if not optimizer else (
        total_loss / len(loader), 100 * correct / total)


def main():
    SR, MAX_LEN, BS, EPOCHS = 16000, 64000, 12, 50

    df_train, df_val = load_dataset('./xeno_dataset/whole_dataset/train_val/train_data'), load_dataset(
        './xeno_dataset/whole_dataset/train_val/validation_data')
    cls_label = {s: i for i, s in enumerate(sorted(df_train['class_name'].unique()))}

    for df in [df_train, df_val]:
        df['class_label'] = df['class_name'].map(cls_label)
        df['audio'] = df['path'].apply(lambda x: get_audio(x, MAX_LEN, SR))
    df_train, df_val = df_train.dropna(subset=['audio']), df_val.dropna(subset=['audio'])

    extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960", return_attention_mask=True)
    processed = DatasetDict({
        'train': HFDataset.from_pandas(df_train),
        'val': HFDataset.from_pandas(df_val)
    }).map(lambda ex: extractor(ex["audio"], sampling_rate=SR, max_length=MAX_LEN, truncation=True, padding=True,
                                return_attention_mask=True),
           remove_columns=["audio", "path", "class_name"], batched=True, batch_size=8,
           num_proc=2 if os.name != 'nt' else 1)

    train_data, val_data = processed["train"].shuffle(seed=42).with_format("numpy")[:], processed["val"].shuffle(
        seed=42).with_format("numpy")[:]
    class_weights = torch.FloatTensor(
        compute_class_weight('balanced', classes=np.unique(train_data['class_label']), y=train_data['class_label'])).to(
        device)

    nw = 0 if os.name == 'nt' else 2
    train_loader = DataLoader(AudioDataset(train_data), BS, True, num_workers=nw, pin_memory=use_amp, drop_last=True)
    val_loader = DataLoader(AudioDataset(val_data), BS, False, num_workers=nw, pin_memory=use_amp, drop_last=True)

    model = HubertForAudioClassification(len(cls_label)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 5, verbose=True)
    scaler = GradScaler() if use_amp else None

    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc']}
    best_acc, patience = 0, 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        for k, v in zip(history.keys(), [train_loss, train_acc, val_loss, val_acc]):
            history[k].append(v)

        print(
            f'Epoch {epoch + 1}/{EPOCHS} - Train: {train_loss:.4f}/{train_acc:.2f}% - Val: {val_loss:.4f}/{val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_Hubert_model.pth')
            patience = 0
        elif (patience := patience + 1) >= 30:
            break

    model.load_state_dict(torch.load('best_Hubert_model.pth'))
    final_loss, final_acc, final_preds, final_labels = run_epoch(model, val_loader, criterion)
    final_f1 = f1_score(final_labels, final_preds, average='weighted')

    print(f"\n最终准确率: {final_acc:.2f}%, F1: {final_f1:.4f}")
    print("\n" + classification_report(final_labels, final_preds,
                                       target_names=[s for s, _ in sorted(cls_label.items(), key=lambda x: x[1])]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (m1, m2), ylabel in zip(axes, [('train_acc', 'val_acc'), ('train_loss', 'val_loss')],
                                    ['Accuracy (%)', 'Loss']):
        ax.plot(history[m1], label=m1.split('_')[0])
        ax.plot(history[m2], label=m2.split('_')[0])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hubert_training_history.png', dpi=300)

    torch.save(model.state_dict(), 'hubert_final.pth')
    with open('hubert_large_config.json', 'w', encoding='utf-8') as f:
        json.dump({"model_type": "HuBERT", "num_classes": len(cls_label), "cls_label": cls_label,
                   "final_accuracy": float(final_acc), "final_f1_score": float(final_f1)}, f, indent=2,
                  ensure_ascii=False)

    if use_amp:
        torch.cuda.empty_cache()
    print(f"\n训练完成，最佳准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()