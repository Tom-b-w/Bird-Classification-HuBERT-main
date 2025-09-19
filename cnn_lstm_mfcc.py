import json
import os
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import random
import warnings
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")


# GPU配置
def configure_gpu():
    """配置GPU设置"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"发现 {len(gpus)} 个GPU设备")
            return True
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")
            return False
    else:
        print("未检测到GPU设备，将使用CPU训练")
        return False


# 设置随机种子
def set_seed(seed=42):
    """设置随机种子确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)


# 数据增强类
class AudioAugmentation:
    """针对鸟鸣的音频数据增强"""

    def __init__(self, sampling_rate=22050):
        self.sr = sampling_rate

    def time_stretch(self, audio, rate_range=(0.9, 1.1)):
        """时间拉伸"""
        try:
            rate = np.random.uniform(*rate_range)
            return librosa.effects.time_stretch(audio, rate=rate)
        except:
            return audio

    def pitch_shift(self, audio, n_steps_range=(-1, 1)):
        """音调变换"""
        try:
            n_steps = np.random.uniform(*n_steps_range)
            return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
        except:
            return audio

    def add_noise(self, audio, noise_factor_range=(0.001, 0.003)):
        """添加白噪声"""
        noise_factor = np.random.uniform(*noise_factor_range)
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise

    def volume_change(self, audio, gain_range=(0.8, 1.2)):
        """音量变化"""
        gain = np.random.uniform(*gain_range)
        return audio * gain


# 数据加载函数
def load_dataset(data_dir, dataset_name="dataset"):
    """加载数据集"""
    label = []
    path = []

    if not os.path.exists(data_dir):
        print(f"警告: 目录不存在 {data_dir}")
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


# 创建类别映射
def create_class_mapping(df_train):
    """根据训练数据创建类别映射"""
    unique_species = sorted(df_train['class_name'].unique())
    cls_label = {species: idx for idx, species in enumerate(unique_species)}
    print(f"\n发现 {len(unique_species)} 种鸟类:")
    for species, idx in cls_label.items():
        count = len(df_train[df_train['class_name'] == species])
        print(f"  {species}: {idx} (样本数: {count})")
    return cls_label, len(unique_species)


# 改进的MFCC提取函数
def get_mfcc_enhanced(path, augmentation=None, is_training=False, augment_prob=0.3):
    """增强版MFCC特征提取"""
    try:
        y, sr = librosa.load(path, sr=22050)
        y = librosa.effects.preemphasis(y, coef=0.97)

        if is_training and augmentation is not None and np.random.random() < augment_prob:
            if np.random.random() < 0.3:
                try:
                    y = augmentation.time_stretch(y)
                except:
                    pass
            if np.random.random() < 0.2:
                try:
                    y = augmentation.pitch_shift(y)
                except:
                    pass
            if np.random.random() < 0.3:
                y = augmentation.add_noise(y)
            if np.random.random() < 0.2:
                y = augmentation.volume_change(y)

        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        return features.T
    except Exception as e:
        print(f"处理音频失败: {path}, 错误: {e}")
        return None


# 数据平衡处理
def balance_dataset(df, min_samples=50, max_samples=150, seed=42):
    """数据平衡处理"""
    print(f"\n=== 数据平衡处理 ===")
    print(f"最小样本数: {min_samples}, 最大样本数: {max_samples}")

    balanced_data = []
    unique_classes = df['class_name'].unique()

    for class_name in unique_classes:
        class_data = df[df['class_name'] == class_name].copy()
        n_samples = len(class_data)
        if n_samples < min_samples:
            if n_samples > 0:
                repeat_factor = min_samples // n_samples + 1
                class_data = pd.concat([class_data] * repeat_factor, ignore_index=True)
                class_data = class_data.iloc[:min_samples]
                print(f"类别 {class_name}: {n_samples} -> {len(class_data)} (重复采样)")
            else:
                print(f"警告: 类别 {class_name} 无样本，跳过")
                continue
        elif n_samples > max_samples:
            class_data = class_data.sample(n=max_samples, random_state=seed)
            print(f"类别 {class_name}: {n_samples} -> {len(class_data)} (随机采样)")
        else:
            print(f"类别 {class_name}: {n_samples} 个样本 (无需调整)")
        balanced_data.append(class_data)

    balanced_df = pd.concat(balanced_data, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return balanced_df


# 特征提取
def extract_features(df, augmentation, is_training, max_time_steps, n_features, augment_prob=0.0):
    """提取MFCC特征"""
    dataset_type = "训练集" if is_training else "验证集"
    augment_info = f"(增强概率: {augment_prob})" if is_training else "(无增强)"
    print(f"\n=== 提取{dataset_type}特征{augment_info} ===")

    features = []
    labels = []
    failed_count = 0

    for i in tqdm(range(len(df)), desc=f"处理{dataset_type}"):
        audio_path = df['path'].iloc[i]
        label = df['class_label'].iloc[i]
        mfcc = get_mfcc_enhanced(audio_path, augmentation, is_training, augment_prob)
        if mfcc is not None:
            if mfcc.shape[0] < max_time_steps:
                pad_width = max_time_steps - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
            elif mfcc.shape[0] > max_time_steps:
                mfcc = mfcc[:max_time_steps, :]
            if mfcc.shape[1] != n_features:
                mfcc = np.resize(mfcc, (max_time_steps, n_features))
            features.append(mfcc)
            labels.append(label)
        else:
            failed_count += 1

    if failed_count > 0:
        print(f"警告: {failed_count} 个文件处理失败")

    features_array = np.asarray(features)
    labels_array = np.asarray(labels)
    print(f"{dataset_type}特征形状: {features_array.shape}")
    return features_array, labels_array


# 注意力机制层
class AttentionLayer(layers.Layer):
    """注意力机制层"""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

    def get_config(self):
        return super(AttentionLayer, self).get_config()


# 构建模型
def build_cnn_lstm_model(max_time_steps, n_features, num_classes):
    """构建CNN+LSTM+Attention模型"""
    print(f"\n=== 构建模型 ===")
    print(f"输入形状: ({max_time_steps}, {n_features})")
    print(f"输出类别数: {num_classes}")

    tf.keras.backend.clear_session()
    input_layer = layers.Input(shape=(max_time_steps, n_features), name='input_layer')
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(input_layer)
    x = layers.GroupNormalization(groups=8)(x)  # 替换 BatchNormalization
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.GroupNormalization(groups=8)(x)  # 替换 BatchNormalization
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)
    x = layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)
    x = AttentionLayer()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.GroupNormalization(groups=8)(x)  # 替换 BatchNormalization
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.GroupNormalization(groups=8)(x)  # 替换 BatchNormalization
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax', name='output_layer')(x)
    model = Model(inputs=input_layer, outputs=output)
    return model


# 绘制训练历史
def plot_training_history(history, save_path='cnn_lstm_training_history.png'):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    ax1.set_title('模型准确率', fontsize=14)
    ax1.set_xlabel('轮次', fontsize=12)
    ax1.set_ylabel('准确率', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(history.history['loss'], label='训练损失', linewidth=2)
    ax2.plot(history.history['val_loss'], label='验证损失', linewidth=2)
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
    configure_gpu()
    set_seed(42)

    # 模型参数
    MAX_TIME_STEPS = 65
    N_FEATURES = 120  # 40 MFCC + 40 delta + 40 delta2
    EPOCHS = 40
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # 数据路径
    train_dir = './xeno_dataset/whole_dataset/train_val/train_data'
    val_dir = './xeno_dataset/whole_dataset/train_val/validation_data'

    # 检查数据路径
    if not os.path.exists(train_dir):
        print(f"错误: 训练数据目录不存在: {train_dir}")
        return
    if not os.path.exists(val_dir):
        print(f"错误: 验证数据目录不存在: {val_dir}")
        return

    # 加载数据集
    print("=== 加载数据集 ===")
    df_train = load_dataset(train_dir, "训练")
    df_val = load_dataset(val_dir, "验证")

    if df_train.empty or df_val.empty:
        print("错误: 数据集为空，请检查数据路径")
        return

    # 创建类别映射
    cls_label, NUM_CLASSES = create_class_mapping(df_train)

    # 添加数字标签
    def name_to_label(x):
        cls_name = x['class_name']
        return cls_label.get(cls_name, 0)

    df_train['class_label'] = df_train.apply(name_to_label, axis=1)
    df_val['class_label'] = df_val.apply(name_to_label, axis=1)

    # 数据平衡处理
    df_train_balanced = balance_dataset(df_train, min_samples=50, max_samples=150)
    df_val_balanced = balance_dataset(df_val, min_samples=20, max_samples=50)
    print(f"平衡后 - 训练集: {len(df_train_balanced)}, 验证集: {len(df_val_balanced)}")

    # 创建音频增强器
    augmentation = AudioAugmentation(sampling_rate=22050)

    # 提取特征
    X_train, y_train = extract_features(
        df_train_balanced, augmentation, True, MAX_TIME_STEPS, N_FEATURES, 0.4
    )
    X_val, y_val = extract_features(
        df_val_balanced, None, False, MAX_TIME_STEPS, N_FEATURES, 0.0
    )

    if len(X_train) == 0 or len(X_val) == 0:
        print("错误: 特征提取失败")
        return

    # 特征标准化
    print("\n=== 特征标准化 ===")
    mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    std = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    print("特征标准化完成")

    # 计算类别权重
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"类别权重: {class_weight_dict}")

    # 构建模型
    model = build_cnn_lstm_model(MAX_TIME_STEPS, N_FEATURES, NUM_CLASSES)
    model.summary()

    # 编译模型
    print("\n=== 编译模型 ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_cnn_lstm_baseline.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    # 训练模型
    print(f"\n=== 开始训练 ===")
    print(f"训练样本: {len(X_train)}")
    print(f"验证样本: {len(X_val)}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"最大轮数: {EPOCHS}")

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE, drop_remainder=True)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # 绘制训练历史
    plot_training_history(history)

    # 模型评估
    print("\n=== 模型评估 ===")
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"训练准确率: {train_accuracy:.4f}")
    print(f"验证准确率: {val_accuracy:.4f}")

    # 预测和分类报告
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    f1 = f1_score(y_val, y_pred_classes, average='weighted')
    print(f"加权平均F1分数: {f1:.4f}")

    target_names = [species for species, _ in sorted(cls_label.items(), key=lambda x: x[1])]
    print("\n=== 详细分类报告 ===")
    print(classification_report(y_val, y_pred_classes, target_names=target_names))

    # 保存模型和配置
    print("\n=== 保存模型和配置 ===")
    model.save('cnn_lstm_baseline_final.keras')

    config = {
        "model_type": "CNN-LSTM with Attention Baseline",
        "cls_label": cls_label,
        "num_classes": NUM_CLASSES,
        "max_time_steps": MAX_TIME_STEPS,
        "n_features": N_FEATURES,
        "final_train_accuracy": float(train_accuracy),
        "final_val_accuracy": float(val_accuracy),
        "final_f1_score": float(f1),
        "training_params": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "data_augmentation": True,
            "feature_engineering": "MFCC + Delta + Delta2",
            "data_balancing": True,
            "class_weighting": True
        },
        "model_architecture": {
            "conv1d_layers": 2,
            "lstm_layers": 2,
            "attention": True,
            "dense_layers": 2,
            "dropout": True,
            "group_normalization": True
        }
    }

    with open('cnn_lstm_baseline_small_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    np.savez('feature_normalization.npz', mean=mean, std=std)

    print("✓ 模型已保存到: cnn_lstm_baseline_final.keras")
    print("✓ 最佳模型已保存到: best_cnn_lstm_baseline.keras")
    print("✓ 配置已保存到: cnn_lstm_baseline_small_config.json")
    print("✓ 标准化参数已保存到: feature_normalization.npz")

    # 训练总结
    print(f"\n=== CNN-LSTM Baseline 训练完成 ===")
    print(f"支持的鸟类种类: {NUM_CLASSES} 种")
    print(f"最终训练准确率: {train_accuracy:.4f}")
    print(f"最终验证准确率: {val_accuracy:.4f}")
    print(f"最终F1分数: {f1:.4f}")

    # 清理内存
    del X_train, X_val, y_train, y_val
    import gc
    gc.collect()
    print("✓ 内存已清理")


if __name__ == '__main__':
    main()