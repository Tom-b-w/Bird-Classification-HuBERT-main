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
from sklearn.metrics import f1_score, classification_report, accuracy_score
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
        print(f"\n{dataset_name}数据集原始样本分布:")
        sample_counts = df['class_name'].value_counts()
        for species, count in sample_counts.items():
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


# MFCC特征提取函数
def get_mfcc(path, augmentation=None, is_training=False, augment_prob=0.3):
    """提取MFCC特征"""
    try:
        y, sr = librosa.load(path)

        # 如果是训练阶段且提供了数据增强器，则应用数据增强
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

        # 提取MFCC特征
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
        return mfccs.T
    except Exception as e:
        print(f"处理音频失败: {path}, 错误: {e}")
        return None


# 数据洗牌（不进行平衡处理）
def shuffle_dataset(df, seed=42):
    """对数据集进行洗牌，保持原始分布"""
    print(f"\n=== 数据洗牌（保持原始分布）===")
    print("不进行数据平衡处理，保持真实数据分布")

    shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"洗牌后总样本数: {len(shuffled_df)}")

    return shuffled_df


# 特征提取
def extract_features(df, augmentation=None, is_training=False, max_time_steps=65, n_features=40, augment_prob=0.0):
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
        mfcc = get_mfcc(audio_path, augmentation, is_training, augment_prob)
        if mfcc is not None:
            # 调整特征尺寸
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
class attention(layers.Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(attention, self).build(input_shape)

    def call(self, x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x, self.W) + self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

    def get_config(self):
        return super(attention, self).get_config()


# 构建CNN-LSTM模型
def build_simple_cnn_lstm_model(max_time_steps, n_features, num_classes):
    """构建CNN+LSTM+Attention模型用于单标签分类"""
    print(f"\n=== 构建单标签分类模型 ===")
    print(f"输入形状: ({max_time_steps}, {n_features})")
    print(f"输出类别数: {num_classes}")

    tf.keras.backend.clear_session()

    # input
    input_layer = layers.Input(shape=(max_time_steps, n_features), name='input_layer')

    # cnn
    x = layers.Conv1D(128, (3,), activation='relu')(input_layer)
    x = layers.Conv1D(64, (3,), activation='relu')(x)

    # lstm
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = attention()(x)

    # dense
    x = layers.Dense(128, 'relu')(x)
    x = layers.Dense(64, 'relu')(x)

    # output - 使用softmax进行单标签多类分类
    output = layers.Dense(num_classes, 'softmax', name='output_layer')(x)

    model = Model(inputs=input_layer, outputs=output)
    return model


# 绘制训练历史
def plot_training_history(history, save_path='training_history.png'):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    ax1.set_title('单标签分类模型准确率', fontsize=14)
    ax1.set_xlabel('轮次', fontsize=12)
    ax1.set_ylabel('准确率', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(history.history['loss'], label='训练损失', linewidth=2)
    ax2.plot(history.history['val_loss'], label='验证损失', linewidth=2)
    ax2.set_title('单标签分类模型损失', fontsize=14)
    ax2.set_xlabel('轮次', fontsize=12)
    ax2.set_ylabel('损失', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"训练历史图已保存到: {save_path}")


# 单标签预测功能
def predict_single_audio(model, audio_path, cls_label, max_time_steps=65, n_features=40):
    """预测单个音频文件的鸟类种类"""
    try:
        # 提取特征
        mfcc = get_mfcc(audio_path)
        if mfcc is not None:
            mfcc = np.resize(mfcc, (max_time_steps, n_features))
            mfcc = mfcc.reshape((1, max_time_steps, n_features))

            # 预测
            pred_probs = model.predict(mfcc, verbose=0)
            predicted_class = np.argmax(pred_probs[0])
            confidence = float(pred_probs[0][predicted_class])

            # 转换为类别名称
            idx_to_species = {idx: species for species, idx in cls_label.items()}
            predicted_species = idx_to_species[predicted_class]

            return predicted_species, confidence
        else:
            return None, 0.0
    except Exception as e:
        print(f"预测音频失败: {e}")
        return None, 0.0


# 主训练流程
def main():
    """主训练流程 - 单标签鸟类识别（不平衡数据）"""
    print("=== 开始单标签鸟类识别训练（使用原始数据分布）===")
    configure_gpu()
    set_seed(42)

    # 模型参数
    MAX_TIME_STEPS = 65
    N_FEATURES = 40  # MFCC特征
    EPOCHS = 40
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # 数据路径
    train_dir = './xeno_dataset/whole_dataset/train_val/train_data'
    val_dir = './xeno_dataset/whole_dataset/train_val/validation_data'

    # 检查数据路径
    if not os.path.exists(train_dir):
        print(f"错误: 训练数据目录不存在: {train_dir}")
        return None
    if not os.path.exists(val_dir):
        print(f"错误: 验证数据目录不存在: {val_dir}")
        return None

    # 加载数据集
    print("=== 加载数据集 ===")
    df_train = load_dataset(train_dir, "训练")
    df_val = load_dataset(val_dir, "验证")

    if df_train.empty or df_val.empty:
        print("错误: 数据集为空，请检查数据路径")
        return None

    # 创建类别映射
    cls_label, NUM_CLASSES = create_class_mapping(df_train)

    # 添加数字标签
    def name_to_label(x):
        cls_name = x['class_name']
        return cls_label.get(cls_name, 0)

    df_train['class_label'] = df_train.apply(name_to_label, axis=1)
    df_val['class_label'] = df_val.apply(name_to_label, axis=1)

    # 数据洗牌（不进行平衡处理）
    df_train_shuffled = shuffle_dataset(df_train)
    df_val_shuffled = shuffle_dataset(df_val)
    print(f"洗牌后 - 训练集: {len(df_train_shuffled)}, 验证集: {len(df_val_shuffled)}")

    # 创建音频增强器
    augmentation = AudioAugmentation(sampling_rate=22050)

    # 提取特征（加入数据增强）
    X_train, y_train = extract_features(
        df_train_shuffled, augmentation, True, MAX_TIME_STEPS, N_FEATURES, 0.3
    )
    X_val, y_val = extract_features(
        df_val_shuffled, None, False, MAX_TIME_STEPS, N_FEATURES, 0.0
    )

    if len(X_train) == 0 or len(X_val) == 0:
        print("错误: 特征提取失败")
        return None

    print(f"最终训练集形状: {X_train.shape}, 标签: {y_train.shape}")
    print(f"最终验证集形状: {X_val.shape}, 标签: {y_val.shape}")

    # 计算类别权重（用于处理不平衡数据）
    unique_labels = np.unique(y_train)
    if len(unique_labels) > 1:
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=y_train
        )
        class_weight_dict = {int(label): float(weight) for label, weight in zip(unique_labels, class_weights)}
        print(f"类别权重（用于处理不平衡）: {class_weight_dict}")
    else:
        class_weight_dict = None
        print("警告: 只有一个类别，不使用类别权重")

    # 构建模型
    model = build_simple_cnn_lstm_model(MAX_TIME_STEPS, N_FEATURES, NUM_CLASSES)
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
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_single_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    # 训练模型
    print(f"\n=== 开始训练单标签分类模型 ===")
    print(f"训练样本: {len(X_train)}")
    print(f"验证样本: {len(X_val)}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"最大轮数: {EPOCHS}")
    print(f"鸟类种类数: {NUM_CLASSES}")
    print("数据处理策略: 保持原始分布 + 类别权重平衡")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # 绘制训练历史
    plot_training_history(history)

    # 模型评估
    print("\n=== 单标签分类模型评估 ===")
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    print(f"最终训练损失: {train_loss:.4f}")
    print(f"最终训练准确率: {train_accuracy:.4f}")
    print(f"最终验证损失: {val_loss:.4f}")
    print(f"最终验证准确率: {val_accuracy:.4f}")

    # 预测和分类报告
    print("\n=== 计算F1分数和详细分类报告 ===")
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # 计算各种F1分数
    f1_weighted = f1_score(y_val, y_pred_classes, average='weighted')
    f1_macro = f1_score(y_val, y_pred_classes, average='macro')
    f1_micro = f1_score(y_val, y_pred_classes, average='micro')

    print(f"加权F1分数: {f1_weighted:.4f}")
    print(f"宏平均F1分数: {f1_macro:.4f}")
    print(f"微平均F1分数: {f1_micro:.4f}")

    # 详细分类报告
    target_names = [species for species, _ in sorted(cls_label.items(), key=lambda x: x[1])]
    print("\n=== 单标签分类详细报告 ===")
    print(classification_report(y_val, y_pred_classes, target_names=target_names))

    # 保存模型和配置
    print("\n=== 保存模型和配置 ===")
    model.save('single_label_bird_classifier.keras')

    config = {
        "model_type": "Single Label Bird Classification (CNN-LSTM with Attention)",
        "task": "single_label_classification",
        "data_strategy": "original_distribution_with_class_weights",
        "cls_label": cls_label,
        "num_classes": NUM_CLASSES,
        "max_time_steps": MAX_TIME_STEPS,
        "n_features": N_FEATURES,
        "performance_metrics": {
            "final_train_loss": float(train_loss),
            "final_train_accuracy": float(train_accuracy),
            "final_val_loss": float(val_loss),
            "final_val_accuracy": float(val_accuracy),
            "f1_weighted": float(f1_weighted),
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro)
        },
        "training_params": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "feature_type": "MFCC",
            "data_augmentation": True,
            "data_balancing": False,
            "class_weighting": True if class_weight_dict else False
        },
        "model_architecture": {
            "conv1d_layers": 2,
            "lstm_layers": 2,
            "attention": True,
            "dense_layers": 2,
            "output_activation": "softmax"
        }
    }

    with open('single_label_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("✓ 单标签分类模型已保存到: single_label_bird_classifier.keras")
    print("✓ 最佳模型已保存到: best_single_model.keras")
    print("✓ 配置已保存到: single_label_config.json")

    # 测试预测功能
    print("\n=== 测试单标签预测功能 ===")
    if len(df_val) > 0:
        test_audio_path = df_val['path'].iloc[0]
        true_species = df_val['class_name'].iloc[0]
        predicted_species, confidence = predict_single_audio(
            model, test_audio_path, cls_label, MAX_TIME_STEPS, N_FEATURES
        )
        print(f"测试音频路径: {test_audio_path}")
        print(f"真实鸟类: {true_species}")
        print(f"预测鸟类: {predicted_species}")
        print(f"预测置信度: {confidence:.4f}")

    # 最终总结
    print(f"\n=== 单标签鸟类识别训练完成 ===")
    print(f"数据策略: 保持原始分布 + 类别权重")
    print(f"支持识别的鸟类种数: {NUM_CLASSES}")
    print(f"最终训练损失: {train_loss:.4f}")
    print(f"最终训练准确率: {train_accuracy:.4f}")
    print(f"最终验证损失: {val_loss:.4f}")
    print(f"最终验证准确率: {val_accuracy:.4f}")
    print(f"最终F1分数(加权): {f1_weighted:.4f}")

    # 清理内存
    del X_train, X_val, y_train, y_val
    import gc
    gc.collect()
    print("✓ 内存已清理")

    return {
        'model': model,
        'cls_label': cls_label,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'num_classes': NUM_CLASSES
    }


if __name__ == '__main__':
    results = main()
    if results:
        print("\n=== 训练结果摘要 ===")
        print(f"鸟类种数: {results['num_classes']}")
        print(f"验证准确率: {results['val_accuracy']:.4f}")
        print(f"验证F1分数: {results['f1_weighted']:.4f}")
    else:
        print("训练失败，请检查数据和配置")