import os
import librosa
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import json
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Model, AutoFeatureExtractor
from datasets import Dataset as HFDataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings('ignore')


# GPU配置
def configure_gpu():
    """配置GPU设置"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"发现 {device_count} 个GPU设备:")
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_info = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {gpu_name} (内存: {memory_info.total_memory // 1024 ** 3}GB)")

        # 指定使用GPU 1
        gpu_id = 1
        if gpu_id < device_count:
            torch.cuda.set_device(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')
            print(f"使用设备: {device} ({torch.cuda.get_device_name(gpu_id)})")
        else:
            device = torch.device('cuda:0')
            print(f"指定的GPU {gpu_id} 不存在，使用默认GPU 0")

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()
        print("启用混合精度训练 (AMP)")
        return device, True
    else:
        print("警告: 未检测到GPU设备，将使用CPU训练")
        return torch.device('cpu'), False


device, use_amp = configure_gpu()


# 数据加载函数
def load_dataset(data_dir, dataset_name="dataset"):
    """加载数据集"""
    label = []
    path = []
    if not os.path.exists(data_dir):
        print(f"错误: 目录不存在 {data_dir}")
        return pd.DataFrame()
    print(f"扫描目录: {data_dir}")
    for species in os.listdir(data_dir):
        species_folder = os.path.join(data_dir, species)
        if os.path.isdir(species_folder):
            for audio in os.listdir(species_folder):
                if audio.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    audio_path = os.path.join(species_folder, audio).replace("\\", "/")
                    if os.path.isfile(audio_path) and os.path.getsize(audio_path) > 0:
                        path.append(audio_path)
                        label.append(species)
    df = pd.DataFrame({'path': path, 'class_name': label})
    print(f"{dataset_name}数据集大小: {df.shape}")
    if not df.empty:
        print(f"\n{dataset_name}数据集样本预览:")
        sample_counts = df['class_name'].value_counts()
        for species, count in sample_counts.head(5).items():
            print(f"  {species}: {count} 个样本")
    return df


# 动态创建类别映射
def create_class_mapping(df_train):
    """根据训练数据创建类别映射"""
    if df_train.empty:
        raise ValueError("训练数据集为空，无法创建类别映射")
    unique_species = sorted(df_train['class_name'].unique())
    cls_label = {species: idx for idx, species in enumerate(unique_species)}
    print(f"\n发现 {len(unique_species)} 种鸟类:")
    for species, idx in cls_label.items():
        count = len(df_train[df_train['class_name'] == species])
        print(f"  {species}: {idx} (样本数: {count})")
    return cls_label, len(unique_species)


def name_to_label(x, cls_label):
    """将类别名称转换为数字标签"""
    cls_name = x['class_name']
    return cls_label.get(cls_name, 0)


# 优化的音频加载函数
def get_audio(x, max_duration, sampling_rate, max_seq_length):
    """加载音频并进行预处理"""
    path = x['path']
    try:
        y, sr = librosa.load(path, sr=sampling_rate, duration=max_duration)
        if len(y) < max_seq_length:
            y = np.pad(y, (0, max_seq_length - len(y)), mode='constant')
        elif len(y) > max_seq_length:
            y = y[:max_seq_length]
        return y
    except Exception as e:
        print(f"加载音频失败: {path}, 错误: {e}")
        return None


# PyTorch Dataset类
class AudioDataset(Dataset):
    """自定义音频数据集"""

    def __init__(self, input_values, attention_mask, labels):
        self.input_values = torch.FloatTensor(input_values)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_values': self.input_values[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


# Wav2Vec2分类模型
class Wav2Vec2ForAudioClassification(nn.Module):
    """基于Wav2Vec2的音频分类模型"""

    def __init__(self, model_checkpoint, num_classes, hidden_dim):
        super(Wav2Vec2ForAudioClassification, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_checkpoint)

        # 冻结特征提取器
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False

        # 分类头 - Wav2Vec2经典架构
        self.dropout1 = nn.Dropout(0.3)  # Wav2Vec2标准dropout配置
        self.dense1 = nn.Linear(hidden_dim, 512)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(512, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(256, 128)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.final_layer = nn.Linear(128, num_classes)
        self.hidden_dim = hidden_dim

    def _get_feat_extract_output_lengths(self, input_lengths):
        """计算Wav2Vec2特征提取后的输出长度"""

        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1

        # Wav2Vec2特征提取器的卷积层配置
        for kernel_size, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        return input_lengths

    def forward(self, input_values, attention_mask=None):
        # 通过Wav2Vec2编码器
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # 注意力掩码处理
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1)
            feature_lengths = self._get_feat_extract_output_lengths(input_lengths)
            batch_size, seq_len, hidden_dim = hidden_states.shape

            # 创建特征级别的注意力掩码
            feature_attention_mask = torch.zeros((batch_size, seq_len),
                                                 device=hidden_states.device, dtype=torch.bool)
            for i, length in enumerate(feature_lengths):
                length = min(length.item(), seq_len)
                feature_attention_mask[i, :length] = True

            # 应用掩码并进行平均池化
            mask = feature_attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
            masked_hidden = hidden_states * mask
            pooled_state = masked_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            # 全局平均池化
            pooled_state = torch.mean(hidden_states, dim=1)

        # 通过分类头
        x = self.dropout1(pooled_state)
        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)

        x = self.dropout3(x)
        x = self.dense3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)

        logits = self.final_layer(x)
        return logits


# 训练函数
def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        input_values = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(input_values, attention_mask)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_values, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


# 验证函数
def validate_epoch(model, test_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_values, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, all_predictions, all_labels


# 绘制训练历史
def plot_training_history(history, save_path='wav2vec2_training_history.png'):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history['train_acc'], label='训练准确率', linewidth=2)
    ax1.plot(history['val_acc'], label='验证准确率', linewidth=2)
    ax1.set_title('模型准确率', fontsize=14)
    ax1.set_xlabel('轮次', fontsize=12)
    ax1.set_ylabel('准确率 (%)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_loss'], label='训练损失', linewidth=2)
    ax2.plot(history['val_loss'], label='验证损失', linewidth=2)
    ax2.set_title('模型损失', fontsize=14)
    ax2.set_xlabel('轮次', fontsize=12)
    ax2.set_ylabel('损失', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"训练历史图已保存到: {save_path}")


# 主训练流程
def main():
    """主训练流程"""
    # 模型配置 - Wav2Vec2优化参数
    MAX_DURATION = 4  # 可调节：音频最大持续时间（秒）
    SAMPLING_RATE = 16000  # Wav2Vec2标准采样率
    BATCH_SIZE = 16  # 可调节：Wav2Vec2经典批次大小
    HIDDEN_DIM = 768  # Wav2Vec2-base隐藏层维度
    MAX_SEQ_LENGTH = MAX_DURATION * SAMPLING_RATE
    MAX_EPOCHS = 40  # 可调节：最大训练轮数
    MODEL_CHECKPOINT = "facebook/wav2vec2-base"  # Wav2Vec2预训练模型

    # 数据路径
    train_dir = './xeno_dataset/whole_dataset/train_val/train_data_backup'
    val_dir = './xeno_dataset/whole_dataset/train_val/validation_data_backup'

    # 加载数据集
    print("=== 加载数据集 ===")
    df_train = load_dataset(train_dir, "训练")
    df_val = load_dataset(val_dir, "验证")
    if df_train.empty or df_val.empty:
        print("错误: 数据集为空，请检查数据路径")
        return

    # 创建类别映射
    global cls_label, NUM_CLASSES
    cls_label, NUM_CLASSES = create_class_mapping(df_train)

    # 添加数字标签
    df_train['class_label'] = df_train.apply(lambda x: name_to_label(x, cls_label), axis=1)
    df_val['class_label'] = df_val.apply(lambda x: name_to_label(x, cls_label), axis=1)

    # 加载音频
    print("\n=== 加载音频 ===")
    df_train['audio'] = df_train.apply(lambda x: get_audio(x, MAX_DURATION, SAMPLING_RATE, MAX_SEQ_LENGTH), axis=1)
    df_train = df_train.dropna(subset=['audio'])
    df_val['audio'] = df_val.apply(lambda x: get_audio(x, MAX_DURATION, SAMPLING_RATE, MAX_SEQ_LENGTH), axis=1)
    df_val = df_val.dropna(subset=['audio'])
    print(f"成功加载 - 训练集: {len(df_train)}, 验证集: {len(df_val)}")

    # 创建数据集
    train_ds = HFDataset.from_pandas(df_train)
    val_ds = HFDataset.from_pandas(df_val)
    dataset = DatasetDict({'train': train_ds, 'val': val_ds})
    print(dataset)

    # 特征提取
    print("\n=== 特征提取 ===")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT, return_attention_mask=True)

    def preprocess_function(examples):
        audio_arrays = examples["audio"]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=True,
            return_attention_mask=True,
        )
        return inputs

    processed_dataset = dataset.map(
        preprocess_function,
        remove_columns=["audio", "path", "class_name"],
        batched=True,
        batch_size=16,  # Wav2Vec2标准处理批次
        num_proc=2 if os.name != 'nt' else 1
    )

    # 转换为numpy格式
    train = processed_dataset["train"].shuffle(seed=42).with_format("numpy")[:]
    val = processed_dataset["val"].shuffle(seed=42).with_format("numpy")[:]

    # 计算类别权重
    print("\n=== 计算类别权重 ===")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train['class_label']),
        y=train['class_label']
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"类别权重: {class_weights.tolist()}")

    # 创建DataLoader
    train_dataset = AudioDataset(train['input_values'], train['attention_mask'], train['class_label'])
    val_dataset = AudioDataset(val['input_values'], val['attention_mask'], val['class_label'])
    num_workers = 0 if os.name == 'nt' else 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )

    # 构建模型
    print("\n=== 构建Wav2Vec2模型 ===")
    model = Wav2Vec2ForAudioClassification(MODEL_CHECKPOINT, NUM_CLASSES, HIDDEN_DIM)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    # 配置优化器和损失函数 - Wav2Vec2优化参数
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)  # Wav2Vec2经典学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    scaler = GradScaler() if use_amp else None

    # 训练历史记录
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # 训练循环
    print(f"\n=== 开始训练 ===")
    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"最大轮数: {MAX_EPOCHS}")

    best_val_acc = 0
    patience = 0
    max_patience = 10  # Wav2Vec2标准早停耐心值

    for epoch in range(MAX_EPOCHS):
        print(f'\nEpoch {epoch + 1}/{MAX_EPOCHS}')
        print('-' * 50)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_wav2vec2_model.pth')
            print(f'保存最佳模型 (验证准确率: {val_acc:.2f}%)')
            patience = 0
        else:
            patience += 1

        if patience >= max_patience:
            print(f"早停: 验证准确率在 {max_patience} 轮内没有改善")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_wav2vec2_large_model.pth'))

    # 最终评估
    print("\n=== 模型评估 ===")
    final_val_loss, final_val_acc, final_preds, final_labels = validate_epoch(model, val_loader, criterion, device)
    final_f1 = f1_score(final_labels, final_preds, average='weighted')
    print(f"最终验证准确率: {final_val_acc:.2f}%")
    print(f"最终验证损失: {final_val_loss:.4f}")
    print(f"最终加权F1分数: {final_f1:.4f}")

    # 分类报告
    print("\n=== 分类报告 ===")
    target_names = [species for species, _ in sorted(cls_label.items(), key=lambda x: x[1])]
    print(classification_report(final_labels, final_preds, target_names=target_names))

    # 绘制训练历史
    plot_training_history(history)

    # 保存模型和配置
    torch.save(model.state_dict(), 'wav2vec2_single_label_large_final.pth')
    config = {
        "model_type": "Wav2Vec2 Single Label Classification",
        "cls_label": cls_label,
        "num_classes": NUM_CLASSES,
        "max_duration": MAX_DURATION,
        "sampling_rate": SAMPLING_RATE,
        "model_checkpoint": MODEL_CHECKPOINT,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "hidden_dim": HIDDEN_DIM,
        "final_accuracy": float(final_val_acc),
        "final_loss": float(final_val_loss),
        "final_f1_score": float(final_f1)
    }
    with open('wav2vec2_large_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 清理GPU内存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("GPU内存已清理")

    print(f"\n=== Wav2Vec2 单标签训练完成 ===")
    print(f"模型权重已保存到: wav2vec2_large_final.pth")
    print(f"最佳模型已保存到: best_wav2vec2_large_model.pth")
    print(f"配置已保存到: wav2vec2_large_config.json")
    print(f"支持的鸟类种类: {list(cls_label.keys())}")
    print(f"总类别数: {NUM_CLASSES}")


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()