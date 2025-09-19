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
from transformers import WavLMConfig, WavLMModel, Wav2Vec2FeatureExtractor
from datasets import Dataset as HFDataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import warnings
import random
import seaborn as sns
from torch.nn.utils import spectral_norm
from pathlib import Path
from collections import Counter
import gc
from typing import Dict, List, Tuple, Optional
import logging

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体用于matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Config:
    """配置类，用于更好的参数管理"""
    # 模型参数
    MAX_DURATION = 4
    SAMPLING_RATE = 16000
    BATCH_SIZE = 16
    HIDDEN_DIM = 768
    DROPOUT_RATE = 0.4
    MAX_EPOCHS = 40
    ACCUMULATION_STEPS = 4
    MODEL_CHECKPOINT = "microsoft/wavlm-base-plus"

    # 训练参数
    PATIENCE = 20
    GRADIENT_CLIP_NORM = 0.5
    WEIGHT_DECAY = 0.02
    PRETRAINED_LR = 5e-6
    NEW_LAYERS_LR = 1e-4
    LABEL_SMOOTHING = 0.1
    FOCAL_ALPHA = 1.0
    FOCAL_GAMMA = 2.0

    # 数据路径
    TRAIN_DIR = './xeno_dataset/whole_dataset/train_val/train_data'
    VAL_DIR = './xeno_dataset/whole_dataset/train_val/validation_data'

    @property
    def max_seq_length(self):
        return self.MAX_DURATION * self.SAMPLING_RATE

    @property
    def effective_batch_size(self):
        return self.BATCH_SIZE * self.ACCUMULATION_STEPS


def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_gpu():
    """配置GPU设置"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"发现 {device_count} 个GPU设备:")
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            logger.info(f"  GPU {i}: {gpu_name}")

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()

        device = torch.device('cuda')
        logger.info(f"使用设备: {device}")
        logger.info("启用混合精度训练 (AMP)")
        return device, True
    else:
        logger.warning("未检测到GPU，将使用CPU训练")
        return torch.device('cpu'), False


class AudioDataLoader:
    """处理音频数据加载和预处理的类"""

    @staticmethod
    def load_dataset(data_dir: str, dataset_name: str = "dataset") -> pd.DataFrame:
        """从目录结构加载音频数据集"""
        if not os.path.exists(data_dir):
            logger.warning(f"目录不存在: {data_dir}")
            return pd.DataFrame()

        logger.info(f"扫描目录: {data_dir}")

        data = {'path': [], 'class_name': []}

        for species in os.listdir(data_dir):
            species_folder = os.path.join(data_dir, species)
            if not os.path.isdir(species_folder):
                continue

            for audio_file in os.listdir(species_folder):
                if audio_file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    audio_path = os.path.join(species_folder, audio_file).replace("\\", "/")
                    if os.path.isfile(audio_path) and os.path.getsize(audio_path) > 0:
                        data['path'].append(audio_path)
                        data['class_name'].append(species)

        df = pd.DataFrame(data)
        logger.info(f"{dataset_name}数据集大小: {df.shape}")

        if not df.empty:
            logger.info(f"\n{dataset_name}数据集预览:")
            sample_counts = df['class_name'].value_counts()
            for species, count in sample_counts.head(5).items():
                logger.info(f"  {species}: {count} 个样本")

        return df

    @staticmethod
    def augment_audio(y: np.ndarray, sr: int = 16000) -> List[np.ndarray]:
        """应用音频数据增强"""
        augmented = [y.copy()]

        # 时间拉伸
        if np.random.random() > 0.5:
            try:
                stretch_rate = np.random.uniform(0.8, 1.2)
                y_stretch = librosa.effects.time_stretch(y, rate=stretch_rate)
                augmented.append(y_stretch)
            except Exception as e:
                logger.debug(f"时间拉伸失败: {e}")

        # 音调变化
        if np.random.random() > 0.5:
            try:
                pitch_shift = np.random.randint(-2, 3)
                y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
                augmented.append(y_pitch)
            except Exception as e:
                logger.debug(f"音调变化失败: {e}")

        # 添加白噪声
        if np.random.random() > 0.5:
            noise_factor = np.random.uniform(0.001, 0.005)
            noise = np.random.randn(len(y)) * noise_factor
            augmented.append(y + noise)

        return augmented

    @staticmethod
    def load_audio(path: str, augment: bool = False, max_duration: int = 4,
                   sampling_rate: int = 16000) -> Optional[np.ndarray]:
        """加载和预处理音频文件"""
        max_seq_length = max_duration * sampling_rate

        try:
            y, sr = librosa.load(path, sr=sampling_rate, duration=max_duration)
            y = librosa.effects.preemphasis(y, coef=0.97)
            y, _ = librosa.effects.trim(y, top_db=20)

            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))

            if augment:
                augmented_list = AudioDataLoader.augment_audio(y, sr)
                y = random.choice(augmented_list)

            if len(y) < max_seq_length:
                y = np.pad(y, (0, max_seq_length - len(y)), mode='constant')
            elif len(y) > max_seq_length:
                if augment:
                    start_idx = np.random.randint(0, len(y) - max_seq_length + 1)
                    y = y[start_idx:start_idx + max_seq_length]
                else:
                    y = y[:max_seq_length]

            return y.astype(np.float32)
        except Exception as e:
            logger.error(f"加载音频失败: {path}, 错误: {e}")
            return None


class ClassificationUtils:
    """分类任务的工具函数"""

    @staticmethod
    def create_class_mapping(df_train: pd.DataFrame) -> Tuple[Dict[str, int], int]:
        """创建类别标签映射"""
        unique_species = sorted(df_train['class_name'].unique())
        cls_label = {species: idx for idx, species in enumerate(unique_species)}

        logger.info(f"\n发现 {len(unique_species)} 种鸟类:")
        for species, idx in cls_label.items():
            count = len(df_train[df_train['class_name'] == species])
            logger.info(f"  {species}: {idx} (样本数: {count})")

        return cls_label, len(unique_species)

    @staticmethod
    def name_to_label(class_name: str, cls_label: Dict[str, int]) -> int:
        """将类别名称转换为标签"""
        if class_name in cls_label:
            return cls_label[class_name]
        else:
            logger.warning(f"未知类别: {class_name}，将跳过此样本")
            return -1


class AudioDataset(Dataset):
    """带实时增强的PyTorch音频数据集"""

    def __init__(self, input_values: np.ndarray, attention_mask: np.ndarray,
                 labels: np.ndarray, augment: bool = False):
        self.input_values = torch.FloatTensor(input_values)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_val = self.input_values[idx].clone()

        if self.augment and torch.rand(1) > 0.5:
            if torch.rand(1) > 0.5:
                mask_length = int(len(input_val) * 0.1)
                if mask_length > 0:
                    mask_start = torch.randint(0, max(1, len(input_val) - mask_length), (1,))
                    input_val[mask_start:mask_start + mask_length] = 0

            if torch.rand(1) > 0.5:
                scale = torch.FloatTensor(1).uniform_(0.8, 1.2)
                input_val = input_val * scale

        return {
            'input_values': input_val,
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


class ImprovedWavLMClassifier(nn.Module):
    """改进的基于WavLM的鸟鸣分类器"""

    def __init__(self, model_checkpoint: str, num_classes: int,
                 hidden_dim: int, dropout_rate: float = 0.4):
        super().__init__()
        try:
            self.wavlm = WavLMModel.from_pretrained(model_checkpoint)
            logger.info(f"成功加载预训练模型: {model_checkpoint}")
        except Exception as e:
            logger.warning(f"加载预训练模型失败: {e}")
            config = WavLMConfig(
                hidden_size=hidden_dim,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_dropout=dropout_rate,
                attention_dropout=dropout_rate,
                feat_extract_dropout=0.1,
                layerdrop=0.1,
                mask_time_prob=0.1,
                mask_feature_prob=0.04,
            )
            self.wavlm = WavLMModel(config)

        self._freeze_pretrained_layers()

        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(hidden_dim, 128, kernel_size=k, padding=k // 2)
            for k in [3, 5, 7, 9]
        ])
        self.conv_bns = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(4)])
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(512)
        self.classifier = self._build_classifier(num_classes, dropout_rate)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

    def _freeze_pretrained_layers(self):
        for param in self.wavlm.feature_extractor.parameters():
            param.requires_grad = False
        for i in range(min(4, len(self.wavlm.encoder.layers))):
            for param in self.wavlm.encoder.layers[i].parameters():
                param.requires_grad = False

    def _build_classifier(self, num_classes: int, dropout_rate: float) -> nn.Module:
        return nn.Sequential(
            nn.Dropout(dropout_rate),
            spectral_norm(nn.Linear(512, 256)),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            spectral_norm(nn.Linear(256, 128)),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            spectral_norm(nn.Linear(128, 64)),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(64, num_classes)
        )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1

        conv_configs = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]
        for kernel_size, stride in conv_configs:
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        return input_lengths

    def forward(self, input_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        batch_size, seq_len, _ = hidden_states.shape

        hidden_transposed = hidden_states.transpose(1, 2)
        multi_scale_features = []

        for conv, bn in zip(self.multi_scale_conv, self.conv_bns):
            feature = conv(hidden_transposed)
            feature = bn(feature)
            feature = torch.relu(feature)
            multi_scale_features.append(feature)

        combined_features = torch.cat(multi_scale_features, dim=1).transpose(1, 2)

        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1)
            feature_lengths = self._get_feat_extract_output_lengths(input_lengths)
            feature_attention_mask = torch.zeros(
                (batch_size, seq_len), device=combined_features.device, dtype=torch.bool
            )
            for i, length in enumerate(feature_lengths):
                length = min(length.item(), seq_len)
                feature_attention_mask[i, :length] = True
            attn_mask = ~feature_attention_mask
        else:
            attn_mask = None

        attended_features, _ = self.attention(
            combined_features, combined_features, combined_features,
            key_padding_mask=attn_mask
        )
        attended_features = self.layer_norm(attended_features)

        if attention_mask is not None:
            mask = (~attn_mask).unsqueeze(-1).expand_as(attended_features).float()
            masked_features = attended_features * mask
            pooled_state = masked_features.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled_state = torch.mean(attended_features, dim=1)

        logits = self.classifier(pooled_state)
        return logits


class CombinedLoss(nn.Module):
    """结合标签平滑和焦点损失的组合损失函数"""

    def __init__(self, weight: Optional[torch.Tensor] = None, smoothing: float = 0.1,
                 focal_alpha: float = 1.0, focal_gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.smoothing = smoothing
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        ce_loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        prob = torch.softmax(pred, dim=-1)
        prob_t = prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - prob_t) ** self.focal_gamma
        focal_loss = self.focal_alpha * focal_weight * ce_loss

        if self.weight is not None:
            focal_loss = focal_loss * self.weight[target]

        return focal_loss.mean()


class Trainer:
    """训练和验证逻辑"""

    def __init__(self, model: nn.Module, device: torch.device, config: Config):
        self.model = model
        self.device = device
        self.config = config

    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module,
                    optimizer: optim.Optimizer, scaler: Optional[GradScaler] = None) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_values = batch['input_values'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            if scaler is not None:
                with autocast():
                    outputs = self.model(input_values, attention_mask)
                    loss = criterion(outputs, labels) / self.config.ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   max_norm=self.config.GRADIENT_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = self.model(input_values, attention_mask)
                loss = criterion(outputs, labels) / self.config.ACCUMULATION_STEPS
                loss.backward()
                if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   max_norm=self.config.GRADIENT_CLIP_NORM)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * self.config.ACCUMULATION_STEPS
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def validate_epoch(self, val_loader: DataLoader,
                       criterion: nn.Module) -> Tuple[float, float, List[int], List[int]]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_values = batch['input_values'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_values, attention_mask)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy, all_predictions, all_labels


class Visualizer:
    """可视化工具"""

    @staticmethod
    def plot_training_history(history: Dict[str, List[float]],
                              save_path: str = 'training_history.png'):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        ax1.plot(history['train_acc'], label='训练准确率', linewidth=2, color='blue')
        ax1.plot(history['val_acc'], label='验证准确率', linewidth=2, color='red')
        ax1.set_title('模型准确率', fontsize=14)
        ax1.set_xlabel('轮次', fontsize=12)
        ax1.set_ylabel('准确率 (%)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history['train_loss'], label='训练损失', linewidth=2, color='blue')
        ax2.plot(history['val_loss'], label='验证损失', linewidth=2, color='red')
        ax2.set_title('模型损失', fontsize=14)
        ax2.set_xlabel('轮次', fontsize=12)
        ax2.set_ylabel('损失', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        overfitting_gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
        ax3.plot(overfitting_gap, linewidth=2, color='orange')
        ax3.set_title('过拟合分析 (训练-验证准确率差)', fontsize=14)
        ax3.set_xlabel('轮次', fontsize=12)
        ax3.set_ylabel('准确率差 (%)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='过拟合警戒线')
        ax3.legend()

        ax4.plot(history['val_acc'], linewidth=3, color='green')
        ax4.set_title('验证准确率趋势', fontsize=14)
        ax4.set_xlabel('轮次', fontsize=12)
        ax4.set_ylabel('验证准确率 (%)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        if history['val_acc']:
            best_epoch = np.argmax(history['val_acc'])
            ax4.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7,
                        label=f'最佳轮次: {best_epoch + 1}')
            ax4.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"训练历史图已保存到: {save_path}")

    @staticmethod
    def plot_confusion_matrix(y_true: List[int], y_pred: List[int],
                              target_names: List[str], save_path: str = 'confusion_matrix.png'):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(max(8, len(target_names)), max(6, len(target_names))))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('混淆矩阵', fontsize=16)
        plt.xlabel('预测标签', fontsize=14)
        plt.ylabel('真实标签', fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"混淆矩阵已保存到: {save_path}")


def preprocess_function(examples: Dict, feature_extractor, max_seq_length: int) -> Dict:
    audio_arrays = examples["audio"]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_seq_length,
        truncation=True,
        padding=True,
        return_attention_mask=True,
    )
    return inputs


def validate_and_clean_audio_data(df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
    logger.info(f"\n=== 验证{dataset_name}音频数据 ===")
    none_count = df['audio'].isna().sum()
    if none_count > 0:
        logger.info(f"发现 {none_count} 个无效音频文件，将被移除")
        df_clean = df.dropna(subset=['audio']).copy()
    else:
        df_clean = df.copy()

    valid_audio = []
    valid_indices = []

    for idx, audio in enumerate(df_clean['audio']):
        if audio is not None:
            try:
                audio_array = np.array(audio, dtype=np.float32)
                valid_audio.append(audio_array)
                valid_indices.append(idx)
            except Exception as e:
                logger.debug(f"音频数据转换失败，索引 {idx}: {e}")

    df_clean = df_clean.iloc[valid_indices].copy()
    df_clean['audio'] = valid_audio
    logger.info(f"{dataset_name}数据验证完成: {len(df_clean)}/{len(df)} 个有效样本")
    return df_clean


def main():
    set_seed(42)
    config = Config()
    device, use_amp = configure_gpu()

    if not os.path.exists(config.TRAIN_DIR):
        logger.error(f"训练数据目录不存在: {config.TRAIN_DIR}")
        return
    if not os.path.exists(config.VAL_DIR):
        logger.error(f"验证数据目录不存在: {config.VAL_DIR}")
        return

    logger.info("=== 加载数据集 ===")
    audio_loader = AudioDataLoader()
    df_train = audio_loader.load_dataset(config.TRAIN_DIR, "训练")
    df_val = audio_loader.load_dataset(config.VAL_DIR, "验证")

    if df_train.empty or df_val.empty:
        logger.error("数据集为空，请检查数据路径")
        return

    cls_utils = ClassificationUtils()
    cls_label, num_classes = cls_utils.create_class_mapping(df_train)

    logger.info(f"\n=== 模型配置 ===")
    logger.info(f"  鸟类种数: {num_classes}")
    logger.info(f"  批次大小: {config.BATCH_SIZE}")
    logger.info(f"  梯度累积步数: {config.ACCUMULATION_STEPS}")
    logger.info(f"  有效批次大小: {config.effective_batch_size}")
    logger.info(f"  最大训练轮数: {config.MAX_EPOCHS}")
    logger.info(f"  音频时长: {config.MAX_DURATION}秒")
    logger.info(f"  Dropout率: {config.DROPOUT_RATE}")

    df_train['class_label'] = df_train['class_name'].apply(
        lambda x: cls_utils.name_to_label(x, cls_label)
    )
    df_val['class_label'] = df_val['class_name'].apply(
        lambda x: cls_utils.name_to_label(x, cls_label)
    )

    df_train = df_train[df_train['class_label'] != -1].reset_index(drop=True)
    df_val = df_val[df_val['class_label'] != -1].reset_index(drop=True)

    logger.info(f"过滤未知类别后 - 训练集: {len(df_train)}, 验证集: {len(df_val)}")

    logger.info("\n=== 加载音频数据 ===")
    logger.info("加载训练集音频...")

    def safe_load_audio(row):
        try:
            return audio_loader.load_audio(
                row['path'], augment=True, max_duration=config.MAX_DURATION,
                sampling_rate=config.SAMPLING_RATE
            )
        except Exception as e:
            logger.warning(f"音频加载失败 {row['path']}: {e}")
            return None

    df_train['audio'] = df_train.apply(safe_load_audio, axis=1)

    logger.info("加载验证集音频...")

    def safe_load_audio_val(row):
        try:
            return audio_loader.load_audio(
                row['path'], augment=False, max_duration=config.MAX_DURATION,
                sampling_rate=config.SAMPLING_RATE
            )
        except Exception as e:
            logger.warning(f"音频加载失败 {row['path']}: {e}")
            return None

    df_val['audio'] = df_val.apply(safe_load_audio_val, axis=1)

    df_train = validate_and_clean_audio_data(df_train, "训练集")
    df_val = validate_and_clean_audio_data(df_val, "验证集")

    logger.info(f"最终数据 - 训练集: {len(df_train)}, 验证集: {len(df_val)}")

    if len(df_train) == 0 or len(df_val) == 0:
        logger.error("有效音频数据不足")
        return

    train_ds = HFDataset.from_pandas(df_train)
    val_ds = HFDataset.from_pandas(df_val)

    dataset = DatasetDict({
        'train': train_ds,
        'test': val_ds
    })

    logger.info(f"数据集创建完成: {dataset}")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base",
        return_attention_mask=True
    )

    processed_dataset = dataset.map(
        lambda x: preprocess_function(x, feature_extractor, config.max_seq_length),
        remove_columns=["audio", "path", "class_name"],
        batched=True
    )

    train = processed_dataset["train"].shuffle(seed=42).with_format("numpy")[:]
    test = processed_dataset["test"].shuffle(seed=42).with_format("numpy")[:]

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train['class_label']),
        y=train['class_label']
    )
    class_weights = torch.FloatTensor(class_weights).to(device)

    train_dataset = AudioDataset(
        train['input_values'],
        train['attention_mask'],
        train['class_label'],
        augment=True
    )
    val_dataset = AudioDataset(
        test['input_values'],
        test['attention_mask'],
        test['class_label'],
        augment=False
    )

    num_workers = 0 if os.name == 'nt' else 2
    pin_memory = device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info("\n=== 构建模型 ===")
    model = ImprovedWavLMClassifier(
        config.MODEL_CHECKPOINT,
        num_classes,
        config.HIDDEN_DIM,
        config.DROPOUT_RATE
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数数量: {total_params:,}")
    logger.info(f"可训练参数数量: {trainable_params:,}")

    criterion = CombinedLoss(
        weight=class_weights,
        smoothing=config.LABEL_SMOOTHING,
        focal_alpha=config.FOCAL_ALPHA,
        focal_gamma=config.FOCAL_GAMMA
    )

    pretrained_params = []
    new_params = []

    for name, param in model.named_parameters():
        if 'wavlm' in name:
            pretrained_params.append(param)
        else:
            new_params.append(param)

    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': config.PRETRAINED_LR},
        {'params': new_params, 'lr': config.NEW_LAYERS_LR}
    ], weight_decay=config.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config.PRETRAINED_LR, config.NEW_LAYERS_LR],
        epochs=config.MAX_EPOCHS,
        steps_per_epoch=len(train_loader) // config.ACCUMULATION_STEPS,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    scaler = GradScaler() if use_amp else None

    trainer = Trainer(model, device, config)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    logger.info(f"\n=== 开始训练 ===")
    logger.info(f"设备: {device}")
    logger.info(f"训练样本数: {len(train_dataset)}")
    logger.info(f"验证样本数: {len(val_dataset)}")

    best_val_acc = 0
    best_model_state = None
    patience = 0

    for epoch in range(config.MAX_EPOCHS):
        logger.info(f'\n第 {epoch + 1}/{config.MAX_EPOCHS} 轮')
        logger.info('-' * 50)

        train_loss, train_acc = trainer.train_epoch(
            train_loader, criterion, optimizer, scaler
        )

        val_loss, val_acc, val_preds, val_labels = trainer.validate_epoch(
            val_loader, criterion
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        logger.info(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        logger.info(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')

        current_lr_pretrained = optimizer.param_groups[0]['lr']
        current_lr_new = optimizer.param_groups[1]['lr']
        logger.info(f'学习率 - 预训练层: {current_lr_pretrained:.2e}, 新层: {current_lr_new:.2e}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'cls_label': cls_label,
                'config': config.__dict__
            }, 'best_model.pth')
            logger.info(f'保存最佳模型 (验证准确率: {val_acc:.2f}%)')
            patience = 0
        else:
            patience += 1

        overfitting_gap = train_acc - val_acc
        logger.info(f'过拟合差距: {overfitting_gap:.2f}%')

        if patience >= config.PATIENCE:
            logger.info(f"早停: {config.PATIENCE} 轮内无改善")
            break

        scheduler.step()

        if overfitting_gap > 15 and epoch > 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
            logger.info("检测到过拟合，降低学习率")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    visualizer = Visualizer()
    visualizer.plot_training_history(history, 'training_history.png')

    logger.info("\n=== 模型评估 ===")
    final_val_loss, final_val_acc, final_preds, final_labels = trainer.validate_epoch(
        val_loader, criterion
    )

    logger.info(f"最终验证准确率: {final_val_acc:.2f}%")
    logger.info(f"最终验证损失: {final_val_loss:.4f}")
    logger.info(f"最佳验证准确率: {best_val_acc:.2f}%")

    f1_macro = f1_score(final_labels, final_preds, average='macro')
    f1_micro = f1_score(final_labels, final_preds, average='micro')

    logger.info(f"F1宏平均: {f1_macro:.4f}")
    logger.info(f"F1微平均: {f1_micro:.4f}")

    target_names = [species for species, _ in sorted(cls_label.items(), key=lambda x: x[1])]
    logger.info("\n=== 分类报告 ===")
    print(classification_report(final_labels, final_preds, target_names=target_names, digits=4))

    visualizer.plot_confusion_matrix(final_labels, final_preds, target_names, 'confusion_matrix.png')

    logger.info("\n=== 错误分析 ===")
    errors = []
    for i in range(len(final_labels)):
        if final_labels[i] != final_preds[i]:
            true_species = target_names[final_labels[i]]
            pred_species = target_names[final_preds[i]]
            errors.append((true_species, pred_species))

    error_counter = Counter(errors)
    logger.info("最常见的分类错误:")
    for (true_label, pred_label), count in error_counter.most_common(5):
        logger.info(f"  {true_label} → {pred_label}: {count}次")

    logger.info("\n=== 保存模型和配置 ===")
    torch.save(model.state_dict(), 'final_model.pth')

    model_config = {
        "model_type": "改进的WavLM分类器",
        "model_checkpoint": config.MODEL_CHECKPOINT,
        "cls_label": cls_label,
        "num_classes": num_classes,
        "config": {
            "max_duration": config.MAX_DURATION,
            "sampling_rate": config.SAMPLING_RATE,
            "batch_size": config.BATCH_SIZE,
            "accumulation_steps": config.ACCUMULATION_STEPS,
            "effective_batch_size": config.effective_batch_size,
            "max_epochs": config.MAX_EPOCHS,
            "hidden_dim": config.HIDDEN_DIM,
            "dropout_rate": config.DROPOUT_RATE,
        },
        "results": {
            "final_accuracy": float(final_val_acc),
            "final_loss": float(final_val_loss),
            "best_accuracy": float(best_val_acc),
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
        },
        "training_params": {
            "label_smoothing": config.LABEL_SMOOTHING,
            "focal_loss": True,
            "focal_alpha": config.FOCAL_ALPHA,
            "focal_gamma": config.FOCAL_GAMMA,
            "gradient_clipping": config.GRADIENT_CLIP_NORM,
            "weight_decay": config.WEIGHT_DECAY,
            "pretrained_lr": config.PRETRAINED_LR,
            "new_layers_lr": config.NEW_LAYERS_LR,
            "spectral_norm": True,
            "data_augmentation": True,
            "class_weighting": True
        },
        "model_architecture": {
            "multi_scale_conv": "3,5,7,9核",
            "attention_heads": 8,
            "spectral_normalization": True,
            "layer_normalization": True,
            "dropout_layers": True,
            "batch_normalization": True
        }
    }

    with open('model_config.json', 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)

    logger.info("✓ 最终模型已保存到: final_model.pth")
    logger.info("✓ 最佳模型已保存到: best_model.pth")
    logger.info("✓ 配置已保存到: model_config.json")
    logger.info("✓ 训练历史图已保存到: training_history.png")
    logger.info("✓ 混淆矩阵已保存到: confusion_matrix.png")

    logger.info(f"\n=== 训练完成 ===")
    logger.info(f"支持的鸟类种数: {num_classes}")
    logger.info(f"鸟类列表: {', '.join(target_names)}")
    logger.info(f"最终验证准确率: {final_val_acc:.2f}%")
    logger.info(f"最佳验证准确率: {best_val_acc:.2f}%")
    logger.info(f"F1宏平均分数: {f1_macro:.4f}")
    logger.info(f"F1微平均分数: {f1_micro:.4f}")

    logger.info(f"\n=== 数据集统计 ===")
    logger.info(f"训练样本数: {len(df_train)}")
    logger.info(f"验证样本数: {len(df_val)}")
    logger.info(f"总训练轮数: {epoch + 1}")

    logger.info(f"\n=== 模型改进措施 ===")
    logger.info("1. 多尺度卷积特征提取 (3,5,7,9核)")
    logger.info("2. 谱归一化防止过拟合")
    logger.info("3. 组合损失函数 (标签平滑 + 焦点损失)")
    logger.info("4. 梯度累积增加有效批次大小")
    logger.info("5. 实时数据增强")
    logger.info("6. 差分学习率策略")
    logger.info("7. OneCycleLR调度器")
    logger.info("8. 类别权重平衡")
    logger.info("9. 全面的过拟合监控")
    logger.info("10. 改进的代码结构和错误处理")

    del train_dataset, val_dataset
    gc.collect()

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("✓ GPU内存已清理")

    logger.info("\n改进的WavLM鸟鸣分类模型训练完成！")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()